<!-- =======================
 AgentSentry README (pretty)
======================== -->

<p align="center">
  <img src="assets/logo.png" alt="AgentSentry logo" width="300">
</p>

<p align="center">
  <b>Agentsentry: Mitigating Indirect Prompt Injection in LLM Agents via Temporal Causal Diagnostics and Context Purification</b>
</p>

<p align="center">
  <a href="https://anonymous.4open.science/r/AgentSentry"><img alt="Code" src="https://img.shields.io/badge/Code-anonymous.4open.science-4c1?style=flat-square"></a>
  <img alt="Defense" src="https://img.shields.io/badge/Topic-Indirect%20Prompt%20Injection-orange?style=flat-square">
  <img alt="Agent" src="https://img.shields.io/badge/Setting-Tool--Augmented%20LLM%20Agents-7b2cbf?style=flat-square">
  <img alt="Benchmark" src="https://img.shields.io/badge/Benchmark-AgentDojo-black?style=flat-square">
</p>

---

## 🧾 Abstract

Large language model (LLM) agents increasingly rely on external tools and retrieval systems to autonomously complete complex tasks. However, this design exposes agents to *indirect prompt injection* (IPI), where attacker-controlled context embedded in tool outputs or retrieved content silently steers agent actions away from user intent. Unlike prompt-based attacks, IPI unfolds over multi-turn trajectories, making malicious control difficult to disentangle from legitimate task execution. Existing inference-time defenses primarily rely on heuristic detection and conservative blocking of high-risk actions, which can prematurely terminate workflows or broadly suppress tool usage under ambiguous multi-turn scenarios. We propose **AgentSentry**, a novel inference-time detection and mitigation framework for tool-augmented LLM agents. To the best of our knowledge, AgentSentry is the first inference-time defense to model multi-turn IPI as a **temporal causal takeover**. It localizes takeover points via controlled counterfactual re-executions at tool-return boundaries and enables safe continuation through causally guided context purification that removes attack-induced deviations while preserving task-relevant evidence. We evaluate AgentSentry on the **AgentDojo** benchmark across four task suites, three IPI attack families, and multiple black-box LLMs. AgentSentry eliminates successful attacks and maintains strong utility under attack, achieving an average **Utility Under Attack (UA)** of **74.55%**, improving UA by **20.8 to 33.6** percentage points over the strongest baselines without degrading benign performance.

---
<p align="center">
  <img src="assets/pipeline.png" alt="AgentSentry pipeline for defending against multi-turn IPI" width="980"/>
</p>

<p align="center">
  <em><b>AgentSentry pipeline for defending against multi-turn IPI.</b>
</p>

---


## 🧠 Threat Model

**Untrusted content** (emails, documents, webpages) enters the agent via **tools/retrieval/memory**.  
Injected instructions can survive processing and become **action-dominant**, causing **unauthorized tool invocations** and **data exfiltration** through legitimate APIs.

AgentSentry mitigates this by:
1. diagnosing **boundary-local causal influence** (counterfactual shadow runs), then  
2. applying **causally gated purification** + **minimal action revision** for safe continuation.

---

## 📦 Repository Layout

A recommended structure (adapt as needed):

```text
.
├── assets/                         # Figures and visual assets used in the README (e.g., pipeline.png)
├── configs/                        # Experiment configuration files (suite/model/attack/defense settings)
├── evaluation/                     # Evaluation utilities: metric aggregation and reporting (e.g., ASR/UA/CU/FPR)
├── results/
│   └── agentdojo/                  # Final benchmark outputs organized for paper tables (AgentDojo results)
├── runs/                           # Intermediate run artifacts (per-run logs, cached traces, debugging dumps)
├── scripts/                        # Entry points and helpers to run benchmarks and reproduce analyses
├── visualization/                  # Plotting and figure-generation scripts/notebooks for paper-ready visuals
├── pi_detector.py                  # Main detection module (AgentSentry inference-time diagnostics + gating)
├── agentsentry_profiles.py         # Named fixed operating points used by the released experiments
├── agentsentry_integration.py      # Minimal AgentDojo construction and environment overrides
└── README.md                       # Project overview, reproduction instructions, and artifact pointers
```
---

## ⚙️ Installation

AgentSentry is evaluated on the **AgentDojo** benchmark.

First, clone the [agentdojo](https://github.com/ethz-spylab/agentdojo) repository.

```bash
git clone https://github.com/ethz-spylab/agentdojo
cd agentdojo
```



### Step 1: Add the released mechanism files

Copy `pi_detector.py`, `agentsentry_profiles.py`, and
`agentsentry_integration.py` into
`agentdojo/src/agentdojo/agent_pipeline/`.

The released `rebuttal-semantic` profile is the fixed operating point used by
the current experiments. Its short-window sequential OLS parameters are
`window_size=3`, `delta_l=0`, `delta_m=0`, and `gamma=0.25`. These values are
fixed experimental decision parameters; they were not fitted from a
calibration dataset, and runtime execution does not load a calibration file.

### Step 2: Register AgentSentry in AgentDojo

Add `agentsentry` to the available defense names, then construct the component
in `Pipeline.from_config` using the released integration helper:

```python
from agentdojo.agent_pipeline.agentsentry_integration import build_agentsentry

...

if (config.defense or "").lower().startswith("agentsentry"):
    tools_loop = ToolsExecutionLoop(
        [
            ToolsExecutor(),
            build_agentsentry(llm, profile_name="rebuttal-semantic"),
        ]
    )
    pipeline = cls([system_message_component, init_query_component, llm, tools_loop])
    pipeline.name = f"{llm_name}-{config.defense}"
    return pipeline
```

Configure the model API credentials through the environment used by your
AgentDojo model adapter. Do not place API keys in any repository file.

## Run the experiments
The instructions are the same as in [agentdojo](https://github.com/ethz-spylab/agentdojo).

Here is an example command to run the experiments:
```bash
python -m agentdojo.scripts.benchmark --model QWEN3_MAX_2025_09_23 --defense Agentsentry --attack important_instructions -s workspace
```

## Agentsentry Implementation in Other Repositories

In `pi_detector.py`, we provide the implementation of agentsentry. You can use it in your own projects.

## OpenClaw Experimental Artifacts

We provide two normalized artifact archives derived from our OpenClaw evaluation.

- `agentsentry_batch_results.zip` contains the batch-level experimental outputs generated on OpenClaw.

- `agentsentry_manual_evaluation_cases.zip` contains manually curated OpenClaw evaluation cases for qualitative inspection.

Together, these artifacts summarize the OpenClaw evaluation results and provide supporting materials for reproducibility and manual verification.

## Acknowledgments

We thank the authors of [agentdojo](https://github.com/ethz-spylab/agentdojo) for providing the environment for the experiments.

