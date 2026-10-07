# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in selection of Cake-generated FlashInfer kernels.

``VLLM_CAKE_ROUTES`` is a comma-separated list of route names. With the variable
unset no Cake FlashInfer module is built or loaded and every call keeps its
existing backend. A route is admitted once, host-side, against the installed
FlashInfer; the numerics of an admitted call are FlashInfer's.

Routes:

* ``gdn_prefill`` -- Gated DeltaNet chunked prefill (Qwen3-Next / Qwen3.5 linear
  attention) through ``flashinfer.gdn_prefill.chunk_gated_delta_rule(...,
  backend="cake_gdn")`` instead of FlashInfer's default SM100 backend. FlashInfer
  serves an explicit ``cake_gdn`` request only from its frozen Cake GDN manifest
  and raises instead of falling back, so a layer takes the route only when
  :func:`cake_gdn_prefill_admission` resolves its per-rank head geometry and
  vLLM's state contract to a frozen variant for every batch the scheduler can
  produce (``vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn``).
"""

import importlib
from typing import NamedTuple

import torch

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

GDN_PREFILL_ROUTE = "gdn_prefill"
KNOWN_ROUTES: frozenset[str] = frozenset({GDN_PREFILL_ROUTE})

# The GDN prefill contract vLLM presents to FlashInfer (``fi_chunk_gated_delta_rule``):
# 64-token chunks, the recurrent state passed in and returned as FP32, gates and
# betas present, no state checkpoints and no indexed state.
_GDN_CHUNK_TOKENS = 64
_GDN_IO_DTYPES = {torch.bfloat16: "bfloat16", torch.float16: "float16"}


def cake_routes() -> frozenset[str]:
    """The route names selected by ``VLLM_CAKE_ROUTES`` (unknown names warn once)."""
    names = frozenset(s.strip() for s in envs.VLLM_CAKE_ROUTES.split(",") if s.strip())
    for name in sorted(names - KNOWN_ROUTES):
        logger.warning_once(
            "VLLM_CAKE_ROUTES names an unknown route %r (known: %s); ignored.",
            name,
            ", ".join(sorted(KNOWN_ROUTES)),
        )
    return names & KNOWN_ROUTES


def cake_route_enabled(name: str) -> bool:
    """Whether the Cake route ``name`` is selected by ``VLLM_CAKE_ROUTES``."""
    if name not in KNOWN_ROUTES:
        raise ValueError(f"unknown Cake route {name!r}")
    return name in cake_routes()


class CakeGDNPrefillAdmission(NamedTuple):
    """Result of :func:`cake_gdn_prefill_admission`."""

    admitted: bool
    # Why the route is not admitted, or the frozen variants it resolved to.
    detail: str
    # One prefill sequence count per distinct frozen variant (for warm-up).
    warmup_num_seqs: tuple[int, ...]


def _gdn_prefill_probe_num_seqs(max_num_seqs: int) -> tuple[int, ...]:
    """Every power of two up to ``max_num_seqs``, plus ``max_num_seqs`` itself."""
    bound = max(int(max_num_seqs), 1)
    counts = {1 << i for i in range(bound.bit_length()) if (1 << i) <= bound}
    counts.add(bound)
    return tuple(sorted(counts))


def cake_gdn_prefill_admission(
    *,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    dtype: torch.dtype,
    compute_capability: tuple[int, int] | None,
    max_num_seqs: int,
) -> CakeGDNPrefillAdmission:
    """Resolve vLLM's GDN prefill contract against FlashInfer's frozen manifest.

    Host-side only. ``num_k_heads`` / ``num_v_heads`` are the layer's per-rank
    head counts. FlashInfer picks the physical schedule (DV-split or full-DV) from
    the number of sequences in a call and otherwise ignores the sequence lengths,
    so every power of two up to ``max_num_seqs`` and ``max_num_seqs`` itself are
    resolved; the route is admitted only when all of them map to a frozen variant.
    """
    if head_k_dim != 128 or head_v_dim != 128:
        return CakeGDNPrefillAdmission(
            False,
            f"requires head_k_dim == head_v_dim == 128, got {head_k_dim}/{head_v_dim}",
            (),
        )
    io_dtype = _GDN_IO_DTYPES.get(dtype)
    if io_dtype is None:
        return CakeGDNPrefillAdmission(
            False, f"requires BF16 or FP16 activations, got {dtype}", ()
        )
    if compute_capability is None:
        return CakeGDNPrefillAdmission(False, "unknown device capability", ())
    try:
        cake_gdn = importlib.import_module("flashinfer.jit.cake_gdn")
    except ImportError as exc:
        return CakeGDNPrefillAdmission(
            False, f"flashinfer.jit.cake_gdn is not importable: {exc}", ()
        )

    variants: dict[str, int] = {}
    try:
        arch = cake_gdn.arch_for_compute_capability(*compute_capability)
        for num_seqs in _gdn_prefill_probe_num_seqs(max_num_seqs):
            route = cake_gdn.select_cake_gdn_prefill_variant(
                arch=arch,
                io_dtype=io_dtype,
                state_dtype="float32",
                num_seqs=num_seqs,
                total_seq_len=_GDN_CHUNK_TOKENS * num_seqs,
                max_seq_len=_GDN_CHUNK_TOKENS,
                num_q_heads=num_k_heads,
                num_k_heads=num_k_heads,
                num_v_heads=num_v_heads,
                use_initial_state=True,
                store_final_state=True,
                checkpoint_every_n_tokens=0,
                use_state_indices=False,
                gates_present=True,
            )
            variants.setdefault(route.variant_name, num_seqs)
    except (NotImplementedError, RuntimeError, OSError, ValueError) as exc:
        # CakeGDNUnsupportedError is a NotImplementedError; a missing or
        # mismatched manifest raises FileNotFoundError, RuntimeError or ValueError.
        return CakeGDNPrefillAdmission(False, str(exc), ())
    return CakeGDNPrefillAdmission(
        True,
        "frozen variants " + ", ".join(sorted(variants)),
        tuple(sorted(variants.values())),
    )
