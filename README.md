# AgentSentry

AgentSentry is an inference-time defense for indirect prompt injection in tool-augmented LLM agents. The released artifact provides the final runtime implementation used in the NDSS version, together with two packaged verification result sets.

## Contents

```text
pi_detector.py                 # AgentSentry detector and repaired-continuation runtime
agentsentry_profiles.py        # Released OLS operating profile
agentsentry_integration.py     # Minimal AgentDojo construction helper
google_llm.py                  # Minimal AgentDojo host constant
results/slack_important_qwen3max/
results/travel_tool_knowledge_qwen3max/
PACKAGE_MANIFEST.json
```

The packaged results contain only AgentSentry runs for the final released configuration.

## Released Profile

The default profile is `released-ols`:

```text
window_size = 3
delta_l = 0
delta_m = 0
gamma = 0.25
K = 1
```

It uses four replay attribution scores, OLS temporal aggregation, delayed matching, probe-guided purification, revised continuation, semantic authorization, and effect-support validation.

## Using AgentSentry with AgentDojo

Copy the following files into `agentdojo/agent_pipeline/` in an AgentDojo checkout:

```text
pi_detector.py
agentsentry_profiles.py
agentsentry_integration.py
google_llm.py
```

Then add AgentSentry to the runner's defense registry and construct it with:

```python
from agentdojo.agent_pipeline.agentsentry_integration import build_agentsentry

pipeline = [
    build_agentsentry(llm, profile_name="released-ols"),
    # existing tool-execution and model components
]
```

Configure credentials through environment variables used by your model adapter. Do not store API keys in repository files.

Example environment variables:

```bash
export OPENAI_API_KEY=<your_api_key>
export OPENAI_BASE_URL=<openai_compatible_base_url>
export AGENTSENTRY_PROFILE=released-ols
export AGENTSENTRY_DEBUG_PRINT=0
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
```

Example AgentDojo invocation after registering the defense:

```bash
python -m agentdojo.scripts.benchmark \
  --model QWEN3_MAX_2025_09_23 \
  --defense AgentSentry \
  --agentsentry-profile released-ols \
  --attack important_instructions \
  -s slack \
  --logdir runs/slack_important_qwen3max \
  -f
```

For a single task--injection pair, add `-ut <USER_TASK_ID> -it <INJECTION_TASK_ID>`.

## Packaged Results

```text
results/slack_important_qwen3max/summary.json
results/slack_important_qwen3max/rows.jsonl
results/slack_important_qwen3max/raw_traces/
results/slack_important_qwen3max/collected_result.md
results/travel_tool_knowledge_qwen3max/summary.json
results/travel_tool_knowledge_qwen3max/summary.dedup.json
results/travel_tool_knowledge_qwen3max/rows.jsonl
results/travel_tool_knowledge_qwen3max/raw_traces/
results/travel_tool_knowledge_qwen3max/logs/
```

Summary counts:

```text
Slack / Important Instructions / Qwen3-Max: completed 105/105, attack_success 0/105.
Travel / Tool Knowledge / Qwen3-Max: completed 140/140, attack_success 0/140.
```

## Validation

A quick syntax check can be run with:

```bash
python -m py_compile pi_detector.py agentsentry_profiles.py agentsentry_integration.py google_llm.py
```

The artifact contains no API keys, local machine paths, unrelated defenses, notebooks, paper drafts, or unrelated experiment directories.
