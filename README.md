<!-- =======================
 AgentSentry README (pretty)
======================== -->

<p align="center">
  <img src="assets/logo (2).png" alt="AgentSentry logo" width="520">
</p>



<h1 align="center">AgentSentry</h1>

<p align="center">
  <b>Mitigating Indirect Prompt Injection in LLM Agents via Temporal Causal Diagnostics and Context Purification</b>
</p>

<p align="center">
  <a href="https://anonymous.4open.science/r/AgentSentry"><img alt="Code" src="https://img.shields.io/badge/Code-anonymous.4open.science-4c1?style=flat-square"></a>
  <img alt="Defense" src="https://img.shields.io/badge/Topic-Indirect%20Prompt%20Injection-orange?style=flat-square">
  <img alt="Agent" src="https://img.shields.io/badge/Setting-Tool--Augmented%20LLM%20Agents-7b2cbf?style=flat-square">
  <img alt="Benchmark" src="https://img.shields.io/badge/Benchmark-AgentDojo-black?style=flat-square">
</p>

<p align="center">
  <a href="#-highlights">Highlights</a> ·
  <a href="#-figures">Figures</a> ·
  <a href="#-quickstart">Quickstart</a> ·
  <a href="#-reproducing-results-agentdojo">Reproducing Results</a> ·
  <a href="#-citation">Citation</a>
</p>

---

## ✦ Highlights

- **Problem:** Tool-augmented LLM agents are vulnerable to **indirect prompt injection (IPI)**, where attacker-controlled context embedded in tool outputs/retrieval silently steers actions away from user intent.
- **Key insight:** Multi-turn IPI can be modeled as a **temporal causal takeover**.
- **Method:** **AgentSentry** performs **boundary-anchored counterfactual shadow runs** at **tool-return boundaries** to localize takeover points, then applies **causally gated context purification** to suppress instruction-carrying influence while preserving task evidence.
- **Outcome:** Enables **safe continuation** (policy-compliant) instead of terminating workflows or globally disabling tool use.
- **Results (AgentDojo):** **ASR = 0%**, **UA = 74.55%** on average, improving UA by **+20.8 to +33.6** points over the strongest baselines without degrading benign performance.

---

## 🧾 Abstract

Large language model (LLM) agents increasingly rely on external tools and retrieval systems to autonomously complete complex tasks. However, this design exposes agents to *indirect prompt injection* (IPI), where attacker-controlled context embedded in tool outputs or retrieved content silently steers agent actions away from user intent. Unlike prompt-based attacks, IPI unfolds over multi-turn trajectories, making malicious control difficult to disentangle from legitimate task execution. Existing inference-time defenses primarily rely on heuristic detection and conservative blocking of high-risk actions, which can prematurely terminate workflows or broadly suppress tool usage under ambiguous multi-turn scenarios. We propose **AgentSentry**, a novel inference-time detection and mitigation framework for tool-augmented LLM agents. To the best of our knowledge, AgentSentry is the first inference-time defense to model multi-turn IPI as a **temporal causal takeover**. It localizes takeover points via controlled counterfactual re-executions at tool-return boundaries and enables safe continuation through causally guided context purification that removes attack-induced deviations while preserving task-relevant evidence. We evaluate AgentSentry on the **AgentDojo** benchmark across four task suites, three IPI attack families, and multiple black-box LLMs. AgentSentry eliminates successful attacks and maintains strong utility under attack, achieving an average **Utility Under Attack (UA)** of **74.55%**, improving UA by **20.8 to 33.6** percentage points over the strongest baselines without degrading benign performance.

---

## 🖼 Figures

> Put your two paper figures into `assets/` as `1.png` and `2.png`.

### (1) Representative attack chain of tool-mediated IPI

<p align="center">
  <img src="assets/1.png" alt="Representative attack chain of tool-mediated IPI" width="720"/>
</p>

### (2) AgentSentry pipeline for defending against multi-turn IPI

<p align="center">
  <img src="assets/2.png" alt="AgentSentry pipeline" width="920"/>
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
