from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True)
class AgentSentryProfile:
    """Auditable AgentSentry configuration used by a named evaluation protocol."""

    name: str
    scope: str
    parameter_source: str
    parameters: Mapping[str, Any]
    ablation_parameters: Mapping[str, Any]


def _profile(
    *,
    name: str,
    scope: str,
    parameter_source: str,
    parameters: dict[str, Any],
    ablation_parameters: dict[str, Any],
) -> AgentSentryProfile:
    return AgentSentryProfile(
        name=name,
        scope=scope,
        parameter_source=parameter_source,
        parameters=MappingProxyType(parameters),
        ablation_parameters=MappingProxyType(ablation_parameters),
    )


REBUTTAL_SEMANTIC_PROFILE = _profile(
    name="rebuttal-semantic",
    scope=(
        "Final revised mechanism: structural controlled replays with zero-extra-call probe-guided "
        "semantic projection, opaque references for redacted bytes embedded in retained structured "
        "identifiers, live-policy-confirmed repaired-state replanning, objective authorization, "
        "and effect-support validation; probe-only evidence is recorded for temporal diagnosis but is "
        "committed to sticky projection only after live-policy confirmation; no attack-template regex "
        "or fixed consequential-tool allowlist"
    ),
    parameter_source=(
        "Fixed global operating point used by the full AgentDojo evaluation. These values are applied "
        "directly; the runtime does not load or require a calibration artifact."
    ),
    parameters={
        "window_size": 3,
        "lag": 1,
        "K": 3,
        "B": 0,
        "alpha": 0.05,
        "p0_suff": 0.20,
        "delta_l": 0.0,
        "delta_m": 0.0,
        "suff_min": 0.0,
        "gamma": 0.25,
    },
    ablation_parameters={
        "use_causal_decision": True,
        "equivalence_outcome": True,
        "use_llm_purify": False,
        "use_llm_revise": True,
        "use_llm_authorization": True,
        "use_llm_effect_validation": True,
        "semantic_mode": True,
        "strict_semantic_pipeline": True,
        "projection_strategy": "probe_guided",
        "use_provenance_gate": False,
        "purify_verify": False,
        "block_if_unpurifiable": True,
        "gate_on_consequential": False,
        "record_overhead": True,
        "collapse_deterministic_replicates": True,
        "adaptive_probe_replay": True,
    },
)


V7_TOKEN_EFFICIENT_PROFILE = _profile(
    name="v7-token-efficient",
    scope=(
        "Token-efficient v7 variant: preserves the same four formally defined replay regimes, "
        "probe-guided projection, temporal diagnosis, and fail-closed policy while combining "
        "objective authorization with effect-support validation, compacting replay-only assistant "
        "history, reusing identity-projection outcomes, and deferring Revise/Finalize until a "
        "commitment check rejects the projected continuation"
    ),
    parameter_source=(
        "Uses the same fixed global operating point directly. This isolated efficiency profile does "
        "not load or require a calibration artifact."
    ),
    parameters=dict(REBUTTAL_SEMANTIC_PROFILE.parameters),
    ablation_parameters={
        **dict(REBUTTAL_SEMANTIC_PROFILE.ablation_parameters),
        "combined_commitment_validation": True,
        "compact_replay_context": True,
        "defer_revise_until_commitment_rejection": True,
    },
)


SEMANTIC_RESEARCH_PROFILE = _profile(
    name="semantic-research",
    scope=(
        "Research-only replay-attribution path with zero-extra-call probe-guided semantic projection; "
        "no attack-template regex, lexical provenance gate, authorization gate, or fail-closed rule"
    ),
    parameter_source=(
        "Fixed research operating point. This profile isolates replay attribution and probe-guided "
        "purification; it is not a deployment policy."
    ),
    parameters={
        "window_size": 3,
        "lag": 1,
        "K": 3,
        "B": 0,
        "alpha": 0.05,
        "p0_suff": 0.20,
        "delta_l": 0.0,
        "delta_m": 0.0,
        "suff_min": 0.0,
        "gamma": 0.25,
    },
    ablation_parameters={
        "use_causal_decision": True,
        "equivalence_outcome": True,
        "use_llm_purify": False,
        "use_llm_revise": True,
        "use_llm_authorization": False,
        "use_llm_effect_validation": False,
        "semantic_mode": True,
        "strict_semantic_pipeline": False,
        "projection_strategy": "probe_guided",
        "use_provenance_gate": False,
        "purify_verify": False,
        "block_if_unpurifiable": False,
        "gate_on_consequential": False,
        "record_overhead": True,
        "collapse_deterministic_replicates": True,
        "adaptive_probe_replay": True,
    },
)


AGENTSENTRY_PROFILES = MappingProxyType(
    {
        REBUTTAL_SEMANTIC_PROFILE.name: REBUTTAL_SEMANTIC_PROFILE,
        V7_TOKEN_EFFICIENT_PROFILE.name: V7_TOKEN_EFFICIENT_PROFILE,
        SEMANTIC_RESEARCH_PROFILE.name: SEMANTIC_RESEARCH_PROFILE,
    }
)

_PROFILE_ALIASES = {
    "rebuttal": REBUTTAL_SEMANTIC_PROFILE.name,
    "revised": REBUTTAL_SEMANTIC_PROFILE.name,
    "final": REBUTTAL_SEMANTIC_PROFILE.name,
    "semantic-final": REBUTTAL_SEMANTIC_PROFILE.name,
    "token-efficient": V7_TOKEN_EFFICIENT_PROFILE.name,
    "v7-efficient": V7_TOKEN_EFFICIENT_PROFILE.name,
    "semantic": SEMANTIC_RESEARCH_PROFILE.name,
    "research": SEMANTIC_RESEARCH_PROFILE.name,
    "unrestricted": SEMANTIC_RESEARCH_PROFILE.name,
}

_RETIRED_PROFILES = frozenset(
    {
        "submitted",
        "paper",
        "agentdojo-submitted",
        "submitted-agentdojo",
        "agentdojo-rebuttal",
        "rebuttal-deterministic",
        "rebuttal-agentdojo",
    }
)


def get_agentsentry_profile(name: str | None) -> AgentSentryProfile:
    normalized = (name or REBUTTAL_SEMANTIC_PROFILE.name).strip().lower().replace("_", "-")
    if normalized in _RETIRED_PROFILES:
        raise ValueError(
            f"AgentSentry profile '{name}' is archived and disabled; "
            "use 'rebuttal-semantic', 'v7-token-efficient', or 'semantic-research'."
        )
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
        "parameter_source": profile.parameter_source,
        "parameters": dict(profile.parameters),
        "ablation_parameters": dict(profile.ablation_parameters),
    }
