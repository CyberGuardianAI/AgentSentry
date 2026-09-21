<!-- =======================
 AgentSentry README
======================== -->

<p align="center">
  <img src="assets/logo.png" alt="AgentSentry logo" width="300">
</p>

<p align="center">
  <b>AgentSentry: Mitigating Indirect Prompt Injection in LLM Agents via Temporal Causal Diagnostics and Context Purification</b>
</p>

<p align="center">
  <a href="https://anonymous.4open.science/r/AgentSentry"><img alt="Code" src="https://img.shields.io/badge/Code-anonymous.4open.science-4c1?style=flat-square"></a>
  <img alt="Defense" src="https://img.shields.io/badge/Topic-Indirect%20Prompt%20Injection-orange?style=flat-square">
  <img alt="Agent" src="https://img.shields.io/badge/Setting-Tool--Augmented%20LLM%20Agents-7b2cbf?style=flat-square">
  <img alt="Benchmark" src="https://img.shields.io/badge/Benchmark-AgentDojo-black?style=flat-square">
</p>

---

## 🧾 Abstract

Tool-augmented large language model (LLM) agents aggregate dialogue history,
tool outputs, and retrieved content into an accumulated runtime state to
complete multi-step tasks. This execution model exposes agents to indirect
prompt injection (IPI) attacks, where attacker-controlled instructions embedded
in external artifacts are incorporated into the agent state and later influence
tool use. Multi-turn IPI is especially challenging because injected content can
persist across intermediate steps and trigger unauthorized state-changing
actions only after user intent and untrusted content have been merged into the
same context. This creates a *source-attribution problem*: once user-goal
signals and untrusted tool-mediated content are merged, existing inference-time
defenses can often detect suspicious behavior but do not directly distinguish
whether a pending action is supported by the user task or influenced by injected
content. Consequently, they tend to rely on conservative blocking, termination,
or broad filtering, sacrificing utility for security.

We propose **AgentSentry**, an inference-time defense that operationalizes
temporal causal diagnostics through boundary-local counterfactual re-executions
at tool-return boundaries. It summarizes the resulting controlled contrasts as
replay-based attribution scores. When a risky proposal is mediator-sensitive,
AgentSentry purifies the implicated mediator content, removes
instruction-bearing control signals while preserving task-relevant evidence,
and resumes execution from the repaired state.

We evaluate AgentSentry on AgentDojo across four task suites, three IPI attack
families, and three black-box LLM backends. AgentSentry records zero observed
successful attacks in the evaluated cases (\(\mathrm{ASR}=0\%\)) and achieves
an average *Utility Under Attack (UA)* of \(76.10\%\). Compared with the
strongest evaluated baseline, AgentSentry improves average UA by
\(1.84\)--\(7.60\) percentage points, with benign FPRs of
\(2.06\%\)--\(5.15\%\). Additional evaluations examine long-horizon and
adaptive attacks, imperfect purification, runtime overhead, and an OpenClaw
engineering case study in a local-agent runtime. The code for AgentSentry is
available at <https://anonymous.4open.science/r/AgentSentry>.

---

<p align="center">
  <img src="assets/pipeline.png" alt="AgentSentry pipeline for defending against multi-turn IPI" width="980"/>
</p>

<p align="center">
  <em><b>AgentSentry pipeline for defending against multi-turn IPI.</b></em>
</p>

---

## 🧠 Threat Model

**Untrusted content** such as emails, documents, webpages, tool returns, and
retrieved records enters the agent through external tools, retrieval, or memory.
Injected instructions can survive intermediate processing and become
**action-dominant**, causing unauthorized tool invocations, data exfiltration,
destructive actions, or off-goal final commitments through otherwise legitimate
APIs.

AgentSentry mitigates this by:

1. diagnosing **boundary-local mediator influence** through counterfactual
   shadow runs at restored tool-return boundaries;
2. aggregating replay attribution scores across **adjacent boundaries** with
   short-window OLS temporal diagnostics;
3. applying **attribution-gated purification**, revised continuation, and
   effect-support validation for safe task progress.

---

## 📦 Repository Layout

```text
.
|-- assets/                         # README figures
|-- configs/                        # Model and runner configuration helpers
|-- evaluation/                     # Metric aggregation utilities
|-- results/                        # Main-table result data and export script
|   |-- main_table_results.py
|   |-- main_table_results.csv
|   `-- main_table_results.json
|-- runs/                           # Preserved AgentDojo traces and final AgentSentry runs
|   |-- qwen3-max-2025-09-23-agentsentry-slack-important/
|   |-- qwen3-max-2025-09-23-agentsentry-travel-tool-knowledge/
|   `-- qwen3-max-2025-09-23-*/     # Other preserved comparison traces
|-- scripts/                        # Benchmark launch helpers
|-- visualization/                  # Plotting scripts for paper figures
|-- pi_detector.py                  # AgentSentry runtime: four scores, OLS, delayed match, repair
|-- agentsentry_profiles.py         # Released operating profile
|-- agentsentry_integration.py      # Minimal AgentDojo construction helper
|-- google_llm.py                   # Minimal AgentDojo host constant used by the runner
|-- agentsentry_batch_results.zip   # OpenClaw batch-level artifacts
|-- agentsentry_manual_evaluation_cases.zip
|-- PACKAGE_MANIFEST.json
`-- README.md
```

The repository preserves the original benchmark utilities and non-ablation
comparison traces, while replacing the AgentSentry runtime with the final
released implementation and moving the two packaged final AgentSentry
verification runs under `runs/`. It does not include API keys, local machine paths, notebooks,
paper drafts, or obsolete AgentSentry ablation-result directories.

---

## ⚙️ Installation

AgentSentry is evaluated on the
[AgentDojo](https://github.com/ethz-spylab/agentdojo) benchmark.

```bash
git clone https://github.com/ethz-spylab/agentdojo
cd agentdojo
```

### Step 1: Add the released mechanism files

Copy the released mechanism files from this repository into AgentDojo:

```text
pi_detector.py
agentsentry_profiles.py
agentsentry_integration.py
google_llm.py
```

Target directory:

```text
agentdojo/src/agentdojo/agent_pipeline/
```

The released operating point is `released-ols`. Its main decision parameters
are fixed as:

```text
window_size = 3
delta_l = 0
delta_m = 0
gamma = 0.25
K = 1
```

This profile uses four replay attribution scores, OLS temporal aggregation,
delayed matching, probe-guided purification, revised continuation, semantic
authorization, and effect-support validation.

### Step 2: Register AgentSentry in AgentDojo

Add `AgentSentry` to the available defense names in your AgentDojo checkout,
then construct the defense with the released integration helper:

```python
from agentdojo.agent_pipeline.agentsentry_integration import build_agentsentry

...

if (config.defense or "").lower() == "agentsentry":
    tools_loop = ToolsExecutionLoop(
        [
            ToolsExecutor(),
            build_agentsentry(llm, profile_name="released-ols"),
        ]
    )
    pipeline = cls([system_message_component, init_query_component, llm, tools_loop])
    pipeline.name = f"{llm_name}-{config.defense}"
    return pipeline
```

Configure model credentials through environment variables used by your
AgentDojo model adapter. Do **not** place API keys in repository files.

Example environment variables:

```bash
export OPENAI_API_KEY=<your_api_key>
export OPENAI_BASE_URL=<openai_compatible_base_url>
export AGENTSENTRY_PROFILE=released-ols
export AGENTSENTRY_DEBUG_PRINT=0
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
```

---

## ▶️ Reproducing the Packaged AgentSentry Runs

The following commands reproduce the two packaged Qwen3-Max AgentSentry runs
after AgentSentry has been registered in AgentDojo.

### Slack / Important Instructions

```bash
python -m agentdojo.scripts.benchmark \
  --model QWEN3_MAX_2025_09_23 \
  --defense AgentSentry \
  --attack important_instructions \
  -s slack \
  --logdir runs/qwen3-max-2025-09-23-agentsentry-slack-important \
  -f
```

Packaged result:

```text
runs/qwen3-max-2025-09-23-agentsentry-slack-important/summary.json
completed_cases = 105
attack_successes = 0
empirical_asr = 0.0
```

### Travel / Tool Knowledge

```bash
python -m agentdojo.scripts.benchmark \
  --model QWEN3_MAX_2025_09_23 \
  --defense AgentSentry \
  --attack tool_knowledge \
  -s travel \
  --logdir runs/qwen3-max-2025-09-23-agentsentry-travel-tool-knowledge \
  -f
```

Packaged result:

```text
runs/qwen3-max-2025-09-23-agentsentry-travel-tool-knowledge/summary.dedup.json
completed_ok = 140
unique_cases = 140
attack_successes = 0
asr_percent = 0.0
utility_successes = 112
ua_percent = 80.0
```

For a single task--injection pair, add:

```bash
-ut <USER_TASK_ID> -it <INJECTION_TASK_ID>
```

To check the released files before running full benchmarks:

```bash
python -m py_compile pi_detector.py agentsentry_profiles.py agentsentry_integration.py google_llm.py
```

The main aggregate table can be exported from `results/main_table_results.py`:

```bash
python results/main_table_results.py --format markdown
python results/main_table_results.py --format csv --output results/main_table_results.csv
python results/main_table_results.py --format json --output results/main_table_results.json
```

---

## AgentSentry Implementation in Other Repositories

`pi_detector.py` contains the AgentSentry inference-time runtime. Projects that
do not use AgentDojo can adapt the same boundary-local interface: restore a
tool-return boundary, execute the four replay regimes, compute the structural
scores, apply the OLS/delayed-match takeover rule, and continue from the
purified mediator state when takeover is attributed.

---

## OpenClaw Experimental Artifacts

We provide two normalized artifact archives derived from the OpenClaw
evaluation.

- `agentsentry_batch_results.zip` contains the batch-level experimental outputs
  generated on OpenClaw.
- `agentsentry_manual_evaluation_cases.zip` contains manually curated OpenClaw
  evaluation cases for qualitative inspection.

Together, these artifacts summarize the OpenClaw evaluation results and provide
supporting materials for reproducibility and manual verification.

---

## Acknowledgments

We thank the authors of
[AgentDojo](https://github.com/ethz-spylab/agentdojo) for providing the
environment for the experiments.
