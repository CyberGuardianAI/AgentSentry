# AgentSentry v7 Source Release

This directory contains the focused AgentSentry v7 implementation used by the revised Qwen3-Max evaluation. It intentionally excludes unrelated experiment drivers, raw model outputs, and baseline implementations.

## Contents

- `src/agentdojo/agent_pipeline/pi_detector.py`: AgentSentry runtime detector and repair pipeline.
- `src/agentdojo/agent_pipeline/agentsentry_profiles.py`: frozen v7 profiles and operating points.
- `src/agentdojo/agent_pipeline/agentsentry_calibration.py`: calibration utilities.
- `tests/test_agentsentry_profiles.py`: profile-level checks.
- `tests/test_agentsentry_semantic_path.py`: semantic-path and replay checks.
- `SHA256SUMS.txt`: SHA-256 digests for this release.

## Default v7 Profile

The paper-facing profile is `rebuttal-semantic`. It uses the frozen configuration declared in `agentsentry_profiles.py`, including `window_size=3`, `lag=1`, `K=3`, `alpha=0.05`, `delta_min=0`, `suff_min=0`, `cusum_k=0.10`, and `cusum_h=0.50`.

```python
from agentdojo.agent_pipeline.agentsentry_profiles import get_agentsentry_profile
from agentdojo.agent_pipeline.pi_detector import AblationConfig, AgentSentry

profile = get_agentsentry_profile("rebuttal-semantic")
detector = AgentSentry(
    llm,
    **dict(profile.parameters),
    ablation=AblationConfig(**dict(profile.ablation_parameters)),
)
```

Copy the three files under `src/agentdojo/agent_pipeline/` into the corresponding AgentDojo package directory, then instantiate `AgentSentry` in the tool-execution loop. Diagnostic replay proposals are dry runs and must not be sent to the external tool executor.

## Focused Validation

From an AgentDojo checkout containing these files:

```bash
python -m pytest -q tests/test_agentsentry_profiles.py tests/test_agentsentry_semantic_path.py
```

The downloadable archive `artifacts/agentsentry_v7_source.zip` contains the same files.
