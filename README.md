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
├── assets/                      # README figures (1.png, 2.png), optional logo.png
├── agentsentry/                 # core library (diagnostics, purification, enforcement)
│   ├── diagnostics/             # temporal causal diagnostics (ACE/DE/IE, TCDT)
│   ├── purification/            # evidence-form rewriting, mediator suppression
│   ├── policy/                  # action impact categorization, gating rules
│   └── runtime/                 # boundary hooks, caching, shadow-run runner
├── configs/                     # experiment configs (suite/model/attack/defense)
├── scripts/                     # run scripts and helpers
├── evaluation/                  # metrics aggregation (ASR/UA/CU/FPR), plotting
└── README.md
