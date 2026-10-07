# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in selection of Cake-generated FlashInfer kernels by route name.

``VLLM_CAKE_ROUTES`` is a comma-separated list of route names. With the variable
unset no Cake FlashInfer module is built or loaded and every call keeps its
existing backend. A route is admitted only when the installed FlashInfer offers
the Cake backend and the call matches the generated programs; the numerics of
an admitted call are FlashInfer's.

Routes:

* ``gdn_prefill`` -- Gated Delta Net (Qwen3-Next / Qwen3.5) chunked prefill
  through ``flashinfer.gdn_prefill.chunk_gated_delta_rule(...,
  backend="cake_gdn")`` instead of FlashInfer's default GDN prefill kernels;
  decided once per layer, only where vLLM already selects the FlashInfer GDN
  prefill backend (``vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn``).
"""

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

GDN_PREFILL_ROUTE = "gdn_prefill"
KNOWN_ROUTES: frozenset[str] = frozenset({GDN_PREFILL_ROUTE})


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
