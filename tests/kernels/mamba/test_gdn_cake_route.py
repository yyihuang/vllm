# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for the opt-in Cake GDN prefill route
(``VLLM_CAKE_ROUTES=gdn_prefill``)."""

import sys
import types
from types import SimpleNamespace
from typing import Any, Literal
from unittest.mock import MagicMock

import pytest
import torch

import vllm.envs as envs
import vllm.model_executor.custom_op as custom_op_mod
import vllm.utils.flashinfer as fi_utils
from vllm.config import CompilationConfig
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn_mod
from vllm.utils import cake_routes
from vllm.utils.cake_routes import (
    CakeGDNPrefillAdmission,
    cake_gdn_prefill_admission,
)
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


@pytest.mark.parametrize(
    ("nvcc", "found", "missing"),
    [
        ("/cuda/bin/nvcc", {"ninja", "c++"}, []),
        ("/cuda/bin/nvcc", {"ninja", "g++"}, []),
        (None, {"ninja", "c++"}, ["nvcc"]),
        ("/cuda/bin/nvcc", {"c++"}, ["ninja"]),
        ("/cuda/bin/nvcc", {"ninja"}, ["c++"]),
    ],
    ids=["complete", "gxx", "no-nvcc", "no-ninja", "no-cxx"],
)
def test_probe_requires_the_source_build_toolchain(monkeypatch, nvcc, found, missing):
    # The Cake GDN kernels are source-only even when flashinfer-cubin is
    # installed, so the probe needs nvcc, ninja and a C++ compiler.
    monkeypatch.setattr(fi_utils, "_flashinfer_nvcc_path", lambda: nvcc)
    monkeypatch.setattr(
        fi_utils.shutil, "which", lambda name: f"/bin/{name}" if name in found else None
    )
    assert fi_utils._cake_gdn_build_tools_missing() == missing

    monkeypatch.setattr(fi_utils, "has_flashinfer", lambda: True)
    fi_utils.has_flashinfer_cake_gdn_prefill.cache_clear()
    try:
        if missing:
            assert not fi_utils.has_flashinfer_cake_gdn_prefill()
    finally:
        fi_utils.has_flashinfer_cake_gdn_prefill.cache_clear()


@pytest.mark.parametrize("backend", ["flashinfer", "cake_gdn"], ids=["default", "cake"])
def test_fi_wrapper_forwards_the_backend_choice(monkeypatch, backend):
    kernel = MagicMock(return_value=torch.zeros(8, 4, 128))
    fake: Any = types.ModuleType("flashinfer.gdn_prefill")
    fake.chunk_gated_delta_rule = kernel
    monkeypatch.setitem(sys.modules, "flashinfer.gdn_prefill", fake)
    monkeypatch.setitem(sys.modules, "flashinfer", types.ModuleType("flashinfer"))

    q = torch.zeros(1, 8, 4, 128)
    cu_seqlens = torch.tensor([0, 8], dtype=torch.int32)
    out, state = gdn_mod.fi_chunk_gated_delta_rule(
        q=q,
        k=q.clone(),
        v=q.clone(),
        g=torch.zeros(1, 8, 4),
        beta=torch.ones(1, 8, 4),
        initial_state=torch.zeros(1, 4, 128, 128),
        output_final_state=False,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=False,
        backend=backend,
    )

    assert state is None and out.shape == (1, 8, 4, 128)
    assert kernel.call_args.kwargs["backend"] == backend
    passed = kernel.call_args.kwargs["cu_seqlens"]
    if backend == "cake_gdn":
        # Passed through unchanged: FlashInfer resolves the host metadata once
        # per distinct tensor, which every GDN layer of a step then shares.
        assert passed is cu_seqlens
    else:
        assert passed.dtype == torch.int64


# A stand-in for FlashInfer's frozen manifest (``flashinfer.jit.cake_gdn``):
# the DV-split schedule is picked while 2 * num_seqs * num_o_heads <= 148 and
# the full-DV schedule otherwise; each schedule has its own set of frozen
# (HEAD_GROUP_LOG2, IS_GQA, NUM_O_HEADS_LOG2) specializations.
_FAKE_FROZEN = {
    "dvsplit": {(1, 0, 2), (1, 0, 5), (2, 0, 6)},
    "full_dv": {(1, 0, 5), (2, 0, 6)},
}


def _install_fake_cake_gdn(monkeypatch, calls: list[dict[str, Any]]):
    class Unsupported(NotImplementedError):
        pass

    def arch_for_compute_capability(major, minor):
        if (major, minor) not in ((10, 0), (10, 3)):
            raise Unsupported(
                f"Cake GDN supports only SM100a/SM103a, got {major}.{minor}"
            )
        return "sm_100a" if minor == 0 else "sm_103a"

    def select_cake_gdn_prefill_variant(**kw):
        calls.append(kw)
        heads = max(kw["num_q_heads"], kw["num_v_heads"])
        group = heads // min(kw["num_q_heads"], kw["num_v_heads"])
        key = (
            group.bit_length() - 1,
            int(kw["num_q_heads"] >= kw["num_v_heads"]),
            heads.bit_length() - 1,
        )
        regime = "dvsplit" if 2 * kw["num_seqs"] * heads <= 148 else "full_dv"
        if key not in _FAKE_FROZEN[regime]:
            raise Unsupported(f"no exact frozen Cake GDN variant for {regime} {key}")
        return SimpleNamespace(
            route_id=f"flashinfer.gdn_prefill.noncp.{regime}",
            variant_name=f"prefill_{regime}_{key}",
        )

    fake: Any = types.ModuleType("flashinfer.jit.cake_gdn")
    fake.CakeGDNUnsupportedError = Unsupported
    fake.arch_for_compute_capability = arch_for_compute_capability
    fake.select_cake_gdn_prefill_variant = select_cake_gdn_prefill_variant
    jit: Any = types.ModuleType("flashinfer.jit")
    jit.cake_gdn = fake
    monkeypatch.setitem(sys.modules, "flashinfer", types.ModuleType("flashinfer"))
    monkeypatch.setitem(sys.modules, "flashinfer.jit", jit)
    monkeypatch.setitem(sys.modules, "flashinfer.jit.cake_gdn", fake)


def _admission(**over):
    args: dict[str, Any] = dict(
        num_k_heads=16,
        num_v_heads=32,
        head_k_dim=128,
        head_v_dim=128,
        dtype=torch.bfloat16,
        compute_capability=(10, 0),
        max_num_seqs=256,
    )
    args.update(over)
    return cake_gdn_prefill_admission(**args)


def test_admission_resolves_every_batch_regime(monkeypatch):
    calls: list[dict[str, Any]] = []
    _install_fake_cake_gdn(monkeypatch, calls)

    result = _admission()

    assert result.admitted
    # Powers of two up to max_num_seqs, probed with the FP32-state contract.
    assert [c["num_seqs"] for c in calls] == [1, 2, 4, 8, 16, 32, 64, 128, 256]
    assert {c["state_dtype"] for c in calls} == {"float32"}
    assert all(c["use_initial_state"] and c["store_final_state"] for c in calls)
    assert all(c["total_seq_len"] == 64 * c["num_seqs"] for c in calls)
    # One warm-up sequence count per distinct frozen variant: DV-split at 1
    # sequence, full-DV from 4 sequences on (2 * 4 * 32 > 148).
    assert result.warmup_num_seqs == (1, 4)
    assert "prefill_dvsplit_(1, 0, 5)" in result.detail
    assert "prefill_full_dv_(1, 0, 5)" in result.detail


def test_admission_includes_a_non_power_of_two_max_num_seqs(monkeypatch):
    calls: list[dict[str, Any]] = []
    _install_fake_cake_gdn(monkeypatch, calls)
    assert _admission(max_num_seqs=200).admitted
    assert [c["num_seqs"] for c in calls] == [1, 2, 4, 8, 16, 32, 64, 128, 200]


def test_admission_rejects_a_geometry_without_frozen_variants(monkeypatch):
    calls: list[dict[str, Any]] = []
    _install_fake_cake_gdn(monkeypatch, calls)
    # Qwen3-Next at TP2: per-rank 8 K heads / 16 V heads -> (1, 0, 4).
    result = _admission(num_k_heads=8, num_v_heads=16)
    assert not result.admitted
    assert "no exact frozen Cake GDN variant" in result.detail
    assert result.warmup_num_seqs == ()


def test_admission_depends_on_the_scheduler_batch_bound(monkeypatch):
    calls: list[dict[str, Any]] = []
    _install_fake_cake_gdn(monkeypatch, calls)
    # Qwen3-Next at TP8: per-rank 2 K heads / 4 V heads -> (1, 0, 2), DV-split
    # only. Admitted while every batch stays DV-split (2 * 16 * 4 <= 148) ...
    assert _admission(num_k_heads=2, num_v_heads=4, max_num_seqs=16).admitted
    # ... and rejected once the scheduler can produce a full-DV batch.
    result = _admission(num_k_heads=2, num_v_heads=4, max_num_seqs=256)
    assert not result.admitted
    assert "full_dv" in result.detail


@pytest.mark.parametrize(
    ("over", "needle"),
    [
        (dict(head_v_dim=64), "head_k_dim == head_v_dim == 128"),
        (dict(dtype=torch.float32), "BF16 or FP16"),
        (dict(compute_capability=None), "unknown device capability"),
        (dict(compute_capability=(9, 0)), "SM100a/SM103a"),
        (dict(compute_capability=(12, 0)), "SM100a/SM103a"),
    ],
    ids=["head-dim", "fp32-io", "no-capability", "sm90", "sm120"],
)
def test_admission_rejects_contracts_outside_the_cake_kernels(
    monkeypatch, over, needle
):
    calls: list[dict[str, Any]] = []
    _install_fake_cake_gdn(monkeypatch, calls)
    result = _admission(**over)
    assert not result.admitted
    assert needle in result.detail
    assert calls == []


def test_admission_reports_a_missing_cake_module(monkeypatch):
    # A FlashInfer whose ``jit`` package has no ``cake_gdn`` module at all.
    jit: Any = types.ModuleType("flashinfer.jit")
    jit.__path__ = []
    monkeypatch.setitem(sys.modules, "flashinfer", types.ModuleType("flashinfer"))
    monkeypatch.setitem(sys.modules, "flashinfer.jit", jit)
    monkeypatch.delitem(sys.modules, "flashinfer.jit.cake_gdn", raising=False)
    result = _admission()
    assert not result.admitted
    assert "flashinfer.jit.cake_gdn" in result.detail


_GEOMETRY = dict(num_k_heads=16, num_v_heads=32, head_k_dim=128, head_v_dim=128)


@pytest.mark.parametrize(
    ("route_on", "active", "has_cake", "admitted", "geometry", "expect"),
    [
        (True, "flashinfer", True, True, _GEOMETRY, "cake_gdn"),
        (False, "flashinfer", True, True, _GEOMETRY, "flashinfer"),
        (True, "triton", True, True, _GEOMETRY, "flashinfer"),
        (True, "flashinfer", False, True, _GEOMETRY, "flashinfer"),
        (True, "flashinfer", True, False, _GEOMETRY, "flashinfer"),
        (True, "flashinfer", True, True, {}, "flashinfer"),
    ],
    ids=[
        "admitted",
        "route-off",
        "triton-active",
        "no-buildable-backend",
        "manifest-miss",
        "geometry-unknown",
    ],
)
def test_custom_op_decides_the_cake_backend_once(
    monkeypatch, route_on, active, has_cake, admitted, geometry, expect
):
    _routes(monkeypatch, "gdn_prefill" if route_on else None)
    monkeypatch.setattr(gdn_mod, "get_current_vllm_config", MagicMock())
    monkeypatch.setattr(
        gdn_mod, "_resolve_gdn_prefill_backend", lambda cfg: ("auto", active)
    )
    monkeypatch.setattr(gdn_mod, "_log_gdn_backend_decision", lambda *a, **k: None)
    monkeypatch.setattr(gdn_mod, "has_flashinfer_cake_gdn_prefill", lambda: has_cake)
    probe = MagicMock(
        return_value=CakeGDNPrefillAdmission(
            admitted, "probe", (1, 4) if admitted else ()
        )
    )
    monkeypatch.setattr(gdn_mod, "cake_gdn_prefill_admission", probe)
    # CustomOp.__init__ reads the compilation config; a bare one keeps the test
    # independent of device inference (no VllmConfig(), no GPU needed).
    compilation_config = CompilationConfig()
    monkeypatch.setattr(
        custom_op_mod, "get_cached_compilation_config", lambda: compilation_config
    )

    op = gdn_mod.ChunkGatedDeltaRule(**geometry)

    assert op.gdn_prefill_backend == active
    assert op.fi_prefill_backend == expect
    assert op.cake_gdn_warmup_num_seqs == ((1, 4) if expect == "cake_gdn" else ())
    if expect == "cake_gdn" or (
        route_on and active == "flashinfer" and has_cake and geometry
    ):
        # The probe is consulted only when the cheap gates pass, with the
        # per-rank geometry the layer supplied.
        assert probe.call_args.kwargs["num_k_heads"] == 16
        assert probe.call_args.kwargs["num_v_heads"] == 32
    else:
        probe.assert_not_called()
