import pytest

from agentdojo.agent_pipeline.agent_pipeline import ablation_from_defense_name, build_agentsentry_kwargs
from agentdojo.agent_pipeline.agentsentry_profiles import get_agentsentry_profile


@pytest.mark.parametrize(
    "profile_name",
    (
        "submitted",
        "submitted-agentdojo",
        "rebuttal-agentdojo",
        "rebuttal-deterministic",
    ),
)
def test_legacy_rule_purifier_profiles_are_archived(profile_name):
    with pytest.raises(ValueError, match="archived and disabled"):
        get_agentsentry_profile(profile_name)


def test_final_rebuttal_profile_is_semantic_and_default(monkeypatch):
    for name in (
        "AGENTSENTRY_PROFILE",
        "AGENTSENTRY_SEMANTIC_MODE",
        "AGENTSENTRY_USE_PROVENANCE_GATE",
        "AGENTSENTRY_USE_LLM_AUTH",
        "AGENTSENTRY_BLOCK_IF_UNPURIFIABLE",
    ):
        monkeypatch.delenv(name, raising=False)

    assert get_agentsentry_profile(None).name == "rebuttal-semantic"
    assert get_agentsentry_profile("rebuttal").name == "rebuttal-semantic"

    ablation = ablation_from_defense_name("AgentSentry")
    kwargs = build_agentsentry_kwargs(ablation, "rebuttal-semantic")

    assert ablation.semantic_mode is True
    assert kwargs["p0_suff"] == 0.20
    assert kwargs["delta_min"] == 0.0
    assert kwargs["suff_min"] == 0.0
    assert kwargs["cusum_k"] == 0.10
    assert kwargs["cusum_h"] == 0.50
    assert ablation.use_llm_purify is False
    assert ablation.purify_verify is False
    assert ablation.projection_strategy == "probe_guided"
    assert ablation.use_llm_revise is True
    assert ablation.use_llm_authorization is True
    assert ablation.use_llm_effect_validation is True
    assert ablation.use_provenance_gate is False
    assert ablation.collapse_deterministic_replicates is True
    assert ablation.adaptive_probe_replay is True
    assert ablation.block_if_unpurifiable is True
    assert ablation.strict_semantic_pipeline is True


def test_v7_token_efficient_profile_is_isolated(monkeypatch):
    for name in (
        "AGENTSENTRY_COMBINED_COMMITMENT_VALIDATION",
        "AGENTSENTRY_COMPACT_REPLAY_CONTEXT",
        "AGENTSENTRY_DEFER_REVISE_UNTIL_COMMITMENT_REJECTION",
    ):
        monkeypatch.delenv(name, raising=False)

    original = ablation_from_defense_name("AgentSentry")
    build_agentsentry_kwargs(original, "rebuttal-semantic")
    assert original.combined_commitment_validation is False
    assert original.compact_replay_context is False
    assert original.defer_revise_until_commitment_rejection is False

    efficient = ablation_from_defense_name("AgentSentry")
    kwargs = build_agentsentry_kwargs(efficient, "v7-token-efficient")
    assert kwargs["delta_min"] == 0.0
    assert efficient.combined_commitment_validation is True
    assert efficient.compact_replay_context is True
    assert efficient.defer_revise_until_commitment_rejection is True
    assert efficient.semantic_mode is True
    assert efficient.strict_semantic_pipeline is True


def test_final_profile_rejects_incomplete_semantic_pipeline(monkeypatch):
    monkeypatch.setenv("AGENTSENTRY_USE_LLM_AUTH", "0")
    ablation = ablation_from_defense_name("AgentSentry")
    kwargs = build_agentsentry_kwargs(ablation, "rebuttal-semantic")

    from agentdojo.agent_pipeline.pi_detector import AgentSentry

    try:
        AgentSentry(object(), **kwargs)
    except ValueError as exc:
        assert "complete semantic pipeline" in str(exc)
    else:
        raise AssertionError("incomplete final profile was accepted")


def test_final_profile_rejects_non_probe_guided_purify(monkeypatch):
    monkeypatch.setenv("AGENTSENTRY_USE_LLM_PURIFY", "true")
    monkeypatch.setenv("AGENTSENTRY_PROJECTION_STRATEGY", "procedure_preserving")
    ablation = ablation_from_defense_name("AgentSentry")
    kwargs = build_agentsentry_kwargs(ablation, "rebuttal-semantic")

    from agentdojo.agent_pipeline.pi_detector import AgentSentry

    try:
        AgentSentry(object(), **kwargs)
    except ValueError as exc:
        assert "probe_guided_projection" in str(exc)
    else:
        raise AssertionError("hard-rule projection was accepted by the final profile")


def test_environment_override_is_explicit(monkeypatch):
    monkeypatch.setenv("AGENTSENTRY_K", "5")
    monkeypatch.setenv("AGENTSENTRY_TAU_IE", "0.75")
    ablation = ablation_from_defense_name("AgentSentry")
    kwargs = build_agentsentry_kwargs(ablation, "semantic-research")

    assert kwargs["K"] == 5
    assert kwargs["tau_ie"] == 0.75


def test_non_llm_rule_projection_is_rejected(monkeypatch):
    monkeypatch.setenv("AGENTSENTRY_PROJECTION_STRATEGY", "evidence-only")
    ablation = ablation_from_defense_name("AgentSentry")
    with pytest.raises(ValueError, match="archived and disabled"):
        build_agentsentry_kwargs(ablation, "semantic-research")


def test_llm_projection_strategy_remains_an_explicit_ablation(monkeypatch):
    monkeypatch.setenv("AGENTSENTRY_USE_LLM_PURIFY", "true")
    monkeypatch.setenv("AGENTSENTRY_PROJECTION_STRATEGY", "evidence-only")
    ablation = ablation_from_defense_name("AgentSentry")
    build_agentsentry_kwargs(ablation, "semantic-research")
    assert ablation.projection_strategy == "evidence_only"


def test_replay_budget_optimizations_are_explicit(monkeypatch):
    monkeypatch.setenv("AGENTSENTRY_COLLAPSE_DETERMINISTIC_REPLICATES", "false")
    monkeypatch.setenv("AGENTSENTRY_ADAPTIVE_PROBE_REPLAY", "false")
    ablation = ablation_from_defense_name("AgentSentry")
    build_agentsentry_kwargs(ablation, "semantic-research")

    assert ablation.collapse_deterministic_replicates is False
    assert ablation.adaptive_probe_replay is False


def test_semantic_research_profile_disables_hard_commitment_gates(monkeypatch):
    for name in (
        "AGENTSENTRY_SEMANTIC_MODE",
        "AGENTSENTRY_USE_PROVENANCE_GATE",
        "AGENTSENTRY_USE_LLM_AUTH",
        "AGENTSENTRY_BLOCK_IF_UNPURIFIABLE",
    ):
        monkeypatch.delenv(name, raising=False)
    ablation = ablation_from_defense_name("AgentSentry")
    build_agentsentry_kwargs(ablation, "semantic-research")

    assert ablation.semantic_mode is True
    assert ablation.use_llm_purify is False
    assert ablation.purify_verify is False
    assert ablation.projection_strategy == "probe_guided"
    assert ablation.use_llm_revise is True
    assert ablation.use_provenance_gate is False
    assert ablation.use_llm_authorization is False
    assert ablation.use_llm_effect_validation is False
    assert ablation.block_if_unpurifiable is False
