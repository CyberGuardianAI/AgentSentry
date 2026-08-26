"""Minimal AgentSentry integration for the released fixed OLS operating point.

Copy this file together with ``pi_detector.py`` and
``agentsentry_profiles.py`` into ``agentdojo/agent_pipeline``.  The runtime
uses the fixed named profile directly; no calibration dataset or calibration
artifact is loaded.
"""

from __future__ import annotations

import os
from typing import Any

try:  # Package installation.
    from .agentsentry_profiles import get_agentsentry_profile
    from .pi_detector import AblationConfig, AgentSentry
except ImportError:  # Standalone artifact inspection.
    from agentsentry_profiles import get_agentsentry_profile
    from pi_detector import AblationConfig, AgentSentry


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value in {None, ""} else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value in {None, ""} else float(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value in {None, ""}:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def build_agentsentry_kwargs(
    profile_name: str | None = None,
    *,
    ablation: AblationConfig | None = None,
) -> dict[str, Any]:
    """Build the released AgentSentry configuration without data calibration."""

    profile = get_agentsentry_profile(
        profile_name or os.getenv("AGENTSENTRY_PROFILE") or "rebuttal-semantic"
    )
    parameters = dict(profile.parameters)
    ablation = ablation or AblationConfig()

    for field, value in profile.ablation_parameters.items():
        setattr(ablation, field, value)

    ablation.projection_granularity = os.getenv(
        "AGENTSENTRY_PROJECTION_GRANULARITY",
        ablation.projection_granularity,
    ).strip().lower().replace("-", "_")
    if ablation.projection_granularity not in {
        "enclosing_block",
        "exact_span",
        "partial_span",
        "identity",
    }:
        raise ValueError(
            "AGENTSENTRY_PROJECTION_GRANULARITY must be enclosing_block, "
            "exact_span, partial_span, or identity"
        )

    ablation.sanitization_only = _env_bool(
        "AGENTSENTRY_SANITIZATION_ONLY", ablation.sanitization_only
    )
    ablation.debug_print = _env_bool("AGENTSENTRY_DEBUG_PRINT", False)

    return {
        "window_size": _env_int("AGENTSENTRY_WINDOW_SIZE", parameters["window_size"]),
        "lag": _env_int("AGENTSENTRY_LAG", parameters["lag"]),
        "K": _env_int("AGENTSENTRY_K", parameters["K"]),
        "B": _env_int("AGENTSENTRY_B", parameters["B"]),
        "alpha": _env_float("AGENTSENTRY_ALPHA", parameters["alpha"]),
        "p0_suff": _env_float("AGENTSENTRY_P0_SUFF", parameters["p0_suff"]),
        "delta_l": _env_float("AGENTSENTRY_DELTA_L", parameters["delta_l"]),
        "delta_m": _env_float("AGENTSENTRY_DELTA_M", parameters["delta_m"]),
        "suff_min": _env_float("AGENTSENTRY_SUFF_MIN", parameters["suff_min"]),
        "gamma": _env_float("AGENTSENTRY_GAMMA", parameters["gamma"]),
        "paper_mode": _env_bool("AGENTSENTRY_PAPER_MODE", True),
        "ablation": ablation,
    }


def build_agentsentry(llm: Any, profile_name: str | None = None) -> AgentSentry:
    """Instantiate AgentSentry using the named fixed operating point."""

    return AgentSentry(llm, **build_agentsentry_kwargs(profile_name))
