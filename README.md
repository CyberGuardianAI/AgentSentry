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

## AgentSentry v7 and Qwen3-Max Baselines

This focused update publishes only the revised AgentSentry v7 implementation and the newly audited Qwen3-Max results for six additional comparison methods.

- **AgentSentry v7 source:** [`agentsentry_v7/`](agentsentry_v7/) or the downloadable [`artifacts/agentsentry_v7_source.zip`](artifacts/agentsentry_v7_source.zip).
- **Six-method Qwen3-Max results:** [`results/qwen3max_six_baselines/`](results/qwen3max_six_baselines/), including exact counts, 95% Clopper--Pearson intervals, evidence labels, CSV, and JSON.
- **Included comparison methods:** CausalArmor, AttriGuard, IPIGuard, AgentArmor, TraceAegis, and ProvSEEK-adapted.

Source archive SHA-256: `9747b040838578845d3e0dbab26b9ac60c3e8bce2337666a10d2197d774bed9d`.

| Defense | CU | FPR | Important UA/ASR | Tool Knowledge UA/ASR | InjecAgent UA/ASR | Avg. UA/ASR |
|---|---:|---:|---:|---:|---:|---:|
| CausalArmor | 69.07 | 65.98 | 49.62/13.23 | 33.57/21.43 | 77.08/1.39 | 53.43/12.02 |
| AttriGuard | 76.29 | 23.71 | 79.70/4.66 | 68.57/16.43 | 76.39/0.69 | 74.89/7.26 |
| IPIGuard | 77.32 | 39.18 | 64.36/4.96 | 68.57/12.86 | 68.75/4.86 | 67.23/7.56 |
| AgentArmor | 77.32 | >=9.28 | 60.30/0.00 | 26.43/8.57 | 63.19/0.00 | 49.97/2.86 |
| TraceAegis | 65.98 | >=34.02 | 54.29/3.46 | 27.14/5.00 | 61.11/0.00 | 47.51/2.82 |
| ProvSEEK-adapted | 53.61 | >=50.52 | 15.64/4.51 | 29.29/45.71 | 45.14/0.69 | 30.02/16.97 |

All values are percentages. Exact counts and 95% intervals are provided in the linked result directory; `>=` marks FPR lower bounds derived from explicit block markers.

The repository-root `pi_detector.py` and the older installation notes below are retained for the original submitted artifact. Use [`agentsentry_v7/README.md`](agentsentry_v7/README.md) for the revised v7 implementation.

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



### Step 1: Add the interface

In `agentdojo/src/agentdojo/agent_pipeline/agent_pipeline.py`, add the interface of agentsentry.
```python
from agentdojo.agent_pipeline.pi_detector import agentsentry

...

DEFENSES = [
    "tool_filter",
    "transformers_pi_detector",
    "spotlighting_with_delimiting",
    "repeat_user_prompt",
     "melon",
     "melon_aug",
     "taskshiled",
    "agentsentry",
]
```

In `def from_config` function, add the agentsentry detector (similar to `tool_filter` and other defenses).
```python

        if config.defense.startswith("Agentsentry"):
            ab = ablation_from_defense_name(config.defense)


            tools_loop = ToolsExecutionLoop(
                [
                    ToolsExecutor(),
                    Agentsentry(
                        llm,
                        window_size=3,
                        lag=1,
                        K=1,
                        B=0,
                        alpha=0.1,
                        tau_ace=0.2,
                        tau_ie=0.5,
                        gamma=0.6,
                        action_sim_threshold=0.6,
                        paper_mode=True,
                        ablation=ab,
                    ),
                ]
            )
            print("[PIPELINE] defense=", config.defense, "ablation=", ab)

            pipeline = cls([system_message_component, init_query_component, llm, tools_loop])
            pipeline.name = f"{llm_name}-{config.defense}"
            return pipeline

```

## Step 2: Add the agentsentry detector

Please copy the entire `pi_detector.py` to `agentdojo/src/agentdojo/agent_pipeline/pi_detector.py`. Rememeber to fill in the api key for the OpenAI model.
Configure the OpenAI API credentials (PowerShell):
$env:OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
$env:OPENAI_BASE_URL="https:"

Modifications:

1. In `class PromptInjectionDetector`, the `query` function is modified by adding the `is_injection` argument in the `extra_args` dictionary. This is used for stopping the agent's execution when an attack is detected.
2. The `class agentsentry` is added.

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

