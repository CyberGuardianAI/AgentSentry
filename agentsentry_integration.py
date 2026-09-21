"""AgentSentry integration helper for AgentDojo-style runners."""

from __future__ import annotations

import os
from typing import Any

try:
    from .agentsentry_profiles import get_agentsentry_profile
    from .pi_detector import AgentSentry, AgentSentryConfig
except ImportError:
    from agentsentry_profiles import get_agentsentry_profile
    from pi_detector import AgentSentry, AgentSentryConfig


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
    config: AgentSentryConfig | None = None,
) -> dict[str, Any]:
    """Build AgentSentry keyword arguments for the released OLS profile."""

    profile = get_agentsentry_profile(
        profile_name or os.getenv("AGENTSENTRY_PROFILE") or "released-ols"
    )
    parameters = dict(profile.parameters)
    config_params = dict(profile.config_parameters)
    if config is None:
        config = AgentSentryConfig(**config_params)
    else:
        for field, value in config_params.items():
            setattr(config, field, value)

    config.debug_print = _env_bool("AGENTSENTRY_DEBUG_PRINT", config.debug_print)
    config.projection_granularity = os.getenv(
        "AGENTSENTRY_PROJECTION_GRANULARITY",
        config.projection_granularity,
    ).strip().lower().replace("-", "_")

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
        "config": config,
    }


def build_agentsentry(llm: Any, profile_name: str | None = None) -> AgentSentry:
    """Instantiate AgentSentry using the released OLS profile."""

    return AgentSentry(llm, **build_agentsentry_kwargs(profile_name))
