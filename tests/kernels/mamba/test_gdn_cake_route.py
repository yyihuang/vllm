# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for the opt-in Cake GDN prefill route
(``VLLM_CAKE_ROUTES=gdn_prefill``)."""

import sys
import types
from typing import Literal
from unittest.mock import MagicMock

import pytest
import torch

import vllm.envs as envs
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.utils import cake_routes
from vllm.utils.flashinfer import _gdn_prefill_offers_cake_backend

pytestmark = pytest.mark.cpu_test


def _routes(monkeypatch, value: str | None):
    if value is None:
        monkeypatch.delenv("VLLM_CAKE_ROUTES", raising=False)
    else:
        monkeypatch.setenv("VLLM_CAKE_ROUTES", value)
    # envs.__getattr__ may have been wrapped in functools.cache by an engine.
    getattr(envs.__getattr__, "cache_clear", lambda: None)()


def test_routes_unset_selects_nothing(monkeypatch):
    _routes(monkeypatch, None)
    assert cake_routes.cake_routes() == frozenset()
    assert not cake_routes.cake_route_enabled("gdn_prefill")


def test_routes_parse_names_and_ignore_unknown(monkeypatch):
    _routes(monkeypatch, " gdn_prefill ,bogus,")
    assert cake_routes.cake_routes() == {"gdn_prefill"}
    assert cake_routes.cake_route_enabled("gdn_prefill")
    with pytest.raises(ValueError):
        cake_routes.cake_route_enabled("bogus")


def test_probe_requires_a_backend_choice_named_cake_gdn():
    def with_cake(
        q, k, v, *, backend: Literal["auto", "flashinfer", "cake_gdn"] = "auto"
    ):
        pass

    def without_cake(q, k, v, *, backend: Literal["auto", "flashinfer"] = "auto"):
        pass

    def no_backend(q, k, v):
        pass

    assert _gdn_prefill_offers_cake_backend(with_cake)
    assert not _gdn_prefill_offers_cake_backend(without_cake)
    assert not _gdn_prefill_offers_cake_backend(no_backend)
    assert not _gdn_prefill_offers_cake_backend(object())


@pytest.mark.parametrize("backend", ["flashinfer", "cake_gdn"], ids=["default", "cake"])
def test_fi_wrapper_forwards_the_backend_choice(monkeypatch, backend):
    gdn_mod = pytest.importorskip(
        "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"
    )
    kernel = MagicMock(return_value=torch.zeros(8, 4, 128))
    fake = types.ModuleType("flashinfer.gdn_prefill")
    fake.chunk_gated_delta_rule = kernel
    monkeypatch.setitem(sys.modules, "flashinfer.gdn_prefill", fake)
    monkeypatch.setitem(sys.modules, "flashinfer", types.ModuleType("flashinfer"))

    q = torch.zeros(1, 8, 4, 128)
    out, state = gdn_mod.fi_chunk_gated_delta_rule(
        q=q,
        k=q.clone(),
        v=q.clone(),
        g=torch.zeros(1, 8, 4),
        beta=torch.ones(1, 8, 4),
        initial_state=torch.zeros(1, 4, 128, 128),
        output_final_state=False,
        cu_seqlens=torch.tensor([0, 8], dtype=torch.int32),
        use_qk_l2norm_in_kernel=False,
        backend=backend,
    )

    assert state is None and out.shape == (1, 8, 4, 128)
    assert kernel.call_args.kwargs["backend"] == backend
    assert kernel.call_args.kwargs["cu_seqlens"].dtype == torch.int64


@pytest.mark.parametrize(
    ("route_on", "active", "sm100", "has_cake", "expect"),
    [
        (True, "flashinfer", True, True, "cake_gdn"),
        (False, "flashinfer", True, True, "flashinfer"),
        (True, "triton", True, True, "flashinfer"),
        (True, "flashinfer", False, True, "flashinfer"),
        (True, "flashinfer", True, False, "flashinfer"),
    ],
    ids=["admitted", "route-off", "triton-active", "not-sm100", "old-flashinfer"],
)
def test_custom_op_decides_the_cake_backend_once(
    monkeypatch, route_on, active, sm100, has_cake, expect
):
    gdn_mod = pytest.importorskip(
        "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"
    )
    _routes(monkeypatch, "gdn_prefill" if route_on else None)
    monkeypatch.setattr(gdn_mod, "get_current_vllm_config", MagicMock())
    monkeypatch.setattr(
        gdn_mod, "_resolve_gdn_prefill_backend", lambda cfg: ("auto", active)
    )
    monkeypatch.setattr(gdn_mod, "_log_gdn_backend_decision", lambda *a, **k: None)
    monkeypatch.setattr(
        gdn_mod.current_platform,
        "is_device_capability_family",
        lambda major: sm100 and major == 100,
    )
    monkeypatch.setattr(gdn_mod, "has_flashinfer_cake_gdn_prefill", lambda: has_cake)

    with set_current_vllm_config(VllmConfig()):
        op = gdn_mod.ChunkGatedDeltaRule()

    assert op.gdn_prefill_backend == active
    assert op.fi_prefill_backend == expect
