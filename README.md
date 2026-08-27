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

## Abstract

Tool-augmented large language model (LLM) agents aggregate dialogue history, tool outputs, and retrieved content into an accumulated runtime state to complete multi-step tasks. This execution model exposes agents to indirect prompt injection (IPI) attacks, where attacker-controlled instructions embedded in external artifacts are incorporated into the agent state and later influence tool use. Multi-turn IPI is especially challenging because injected content can persist across intermediate steps and trigger unauthorized state-changing actions only after user intent and untrusted content have been merged into the same context. This creates a *source-attribution problem*: once user-goal signals and untrusted tool-mediated content are merged, existing inference-time defenses can often detect suspicious behavior but do not directly distinguish whether a pending action is supported by the user task or influenced by injected content. Consequently, they tend to rely on conservative blocking, termination, or broad filtering, sacrificing utility for security.
We propose **AgentSentry**, an inference-time defense that operationalizes temporal causal diagnostics through boundary-local counterfactual re-executions at tool-return boundaries. It substitutes the mediator under a restored snapshot and records whether the pending proposal survives, aggregating the resulting attribution scores across adjacent boundaries. When a proposal is mediator-sensitive, AgentSentry purifies the implicated mediator content while preserving task-relevant evidence and resumes execution from the repaired state.
We evaluate AgentSentry on AgentDojo across four task suites, three IPI attack families, and three black-box LLM backends. AgentSentry records zero observed successful attacks in the evaluated cases (ASR = 0%) and achieves an average *Utility Under Attack (UA)* of 76.10%. Compared with the strongest evaluated baseline, AgentSentry improves average UA by 1.84–7.60 percentage points, with benign FPRs of 2.06%–5.15%. Additional evaluations examine long-horizon and adaptive attacks, imperfect purification, runtime overhead, and an OpenClaw engineering case study.

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

1. diagnosing **boundary-local mediator influence** through counterfactual shadow runs, and aggregating the resulting scores across **adjacent boundaries**;
2. applying **attribution-gated purification** + **minimal action revision** for safe continuation.
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

