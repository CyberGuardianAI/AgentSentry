from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True)
class AgentSentryProfile:
    """Auditable AgentSentry configuration used by the released protocol."""

    name: str
    scope: str
    calibration_source: str
    parameters: Mapping[str, Any]
    config_parameters: Mapping[str, Any]


def _profile(
    *,
    name: str,
    scope: str,
    calibration_source: str,
    parameters: dict[str, Any],
    config_parameters: dict[str, Any],
) -> AgentSentryProfile:
    return AgentSentryProfile(
        name=name,
        scope=scope,
        calibration_source=calibration_source,
        parameters=MappingProxyType(parameters),
        config_parameters=MappingProxyType(config_parameters),
    )


RELEASED_OLS_PROFILE = _profile(
    name="released-ols",
    scope=(
        "Released AgentSentry mechanism: four controlled-replay attribution scores, "
        "OLS temporal aggregation, delayed action matching, probe-guided purification, "
        "revised continuation, semantic authorization, and effect-support validation."
    ),
    calibration_source=(
        "Frozen OLS operating point used by the released Qwen3-Max evaluations. "
        "Only the released attribution, OLS, purification, revision, and validation path is enabled."
    ),
    parameters={
        "window_size": 3,
        "lag": 1,
        "K": 1,
        "B": 0,
        "alpha": 0.05,
        "p0_suff": 0.20,
        "delta_l": 0.0,
        "delta_m": 0.0,
        "suff_min": 0.0,
        "gamma": 0.25,
    },
    config_parameters={
        "enable_sanitize": True,
        "enable_temporal": True,
        "takeover_sanitize": True,
        "use_llm_purify": False,
        "use_llm_revise": True,
        "use_llm_authorization": True,
        "use_llm_effect_validation": True,
        "semantic_mode": True,
        "strict_semantic_pipeline": True,
        "projection_strategy": "probe_guided",
        "projection_granularity": "enclosing_block",
        "sanitization_only": False,
        "use_provenance_gate": False,
        "purify_verify": False,
        "block_if_unpurifiable": True,
        "gate_on_consequential": False,
        "record_overhead": True,
        "collapse_deterministic_replicates": True,
        "adaptive_probe_replay": True,
    },
)


AGENTSENTRY_PROFILES = MappingProxyType({RELEASED_OLS_PROFILE.name: RELEASED_OLS_PROFILE})

_PROFILE_ALIASES = {
    "default": RELEASED_OLS_PROFILE.name,
    "final": RELEASED_OLS_PROFILE.name,
    "ols": RELEASED_OLS_PROFILE.name,
    "released": RELEASED_OLS_PROFILE.name,
}


def get_agentsentry_profile(name: str | None) -> AgentSentryProfile:
    normalized = (name or RELEASED_OLS_PROFILE.name).strip().lower().replace("_", "-")
    normalized = _PROFILE_ALIASES.get(normalized, normalized)
    try:
        return AGENTSENTRY_PROFILES[normalized]
    except KeyError as exc:
        available = ", ".join(sorted(AGENTSENTRY_PROFILES))
        raise ValueError(f"Unknown AgentSentry profile '{name}'. Available profiles: {available}") from exc


def profile_manifest(profile: AgentSentryProfile) -> dict[str, Any]:
    return {
        "name": profile.name,
        "scope": profile.scope,
        "calibration_source": profile.calibration_source,
        "parameters": dict(profile.parameters),
        "config_parameters": dict(profile.config_parameters),
    }
