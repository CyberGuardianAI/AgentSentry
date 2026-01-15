# AgentSentry: Mitigating Indirect Prompt Injection in LLM Agents via Temporal Causal Diagnostics and Context Purification

Official implementation of **AgentSentry**, an inference-time defense framework for tool-augmented LLM agents against **multi-turn indirect prompt injection (IPI)**.

> **Anonymous code release:** https://anonymous.4open.science/r/AgentSentry

---

## Overview

Tool-augmented LLM agents increasingly rely on external tools and retrieval systems (e.g., search, email, calendar, enterprise APIs) to autonomously complete complex tasks. However, this design exposes agents to **indirect prompt injection (IPI)**, where attacker-controlled content embedded in tool outputs or retrieved artifacts silently steers agent actions away from user intent. Unlike direct prompt attacks, IPI often unfolds over **multi-turn trajectories**, making malicious control difficult to disentangle from legitimate task execution.

Existing inference-time defenses primarily rely on heuristic detection and conservative blocking of high-risk actions, which can prematurely terminate workflows or broadly suppress tool usage under ambiguous multi-turn scenarios.

**AgentSentry** models multi-turn IPI as a **temporal causal takeover**. It:
1. **Localizes takeover points** via controlled **counterfactual re-executions** at **tool-return boundaries**; and  
2. Enables **safe continuation** via **causally guided context purification** that suppresses instruction-carrying influence while preserving task-relevant evidence.

---

## Figures

### Representative attack chain (tool-mediated IPI)

Place this image at `assets/1.png`.

![A representative attack chain of tool-mediated IPI.](assets/1.png)

### AgentSentry pipeline (temporal causal diagnostics + purification + safe continuation)

Place this image at `assets/2.png`.

![AgentSentry pipeline for defending against multi-turn IPI.](assets/2.png)

---

## Abstract

Large language model (LLM) agents increasingly rely on external tools and retrieval systems to autonomously complete complex tasks. However, this design exposes agents to *indirect prompt injection* (IPI), where attacker-controlled context embedded in tool outputs or retrieved content silently steers agent actions away from user intent. Unlike prompt-based attacks, IPI unfolds over multi-turn trajectories, making malicious control difficult to disentangle from legitimate task execution.

Existing inference-time defenses primarily rely on heuristic detection and conservative blocking of high-risk actions, which can prematurely terminate workflows or broadly suppress tool usage under ambiguous multi-turn scenarios. We propose **AgentSentry**, a novel inference-time detection and mitigation framework for tool-augmented LLM agents. To the best of our knowledge, AgentSentry is the first inference-time defense to model multi-turn IPI as a **temporal causal takeover**. It localizes takeover points via controlled counterfactual re-executions at tool-return boundaries and enables safe continuation through causally guided context purification that removes attack-induced deviations while preserving task-relevant evidence.

We evaluate AgentSentry on the **AgentDojo** benchmark across four task suites, three IPI attack families, and multiple black-box LLMs. AgentSentry eliminates successful attacks and maintains strong utility under attack, achieving an average **Utility Under Attack (UA)** of **74.55%**, improving UA by **20.8 to 33.6** percentage points over the strongest baselines without degrading benign performance.

---

## Key Idea

AgentSentry instruments the agent loop at **tool-return boundaries** (immediately after a tool response is incorporated into the evolving context, but before the next action is finalized). At each boundary, AgentSentry runs a small family of **counterfactual shadow executions** by replaying cached mediator content under controlled substitutions to estimate **boundary-local causal influence**. When causal evidence indicates a mediator-driven takeover, AgentSentry **purifies only the newly introduced mediator-bearing content** into an evidence-only form and **minimally revises** the next action to remain policy-compliant, enabling **safe continuation** rather than terminating the workflow.

---

## Results (AgentDojo)

**Headline:** AgentSentry eliminates successful attacks while preserving utility.

- **Attack Success Rate (ASR):** 0% (across evaluated attacks/models in our setting)
- **Utility Under Attack (UA):** 74.55% on average  
- **UA improvement over strongest baselines:** +20.8 to +33.6 points  
- **Benign performance:** no degradation (in our setting)

> For exact numbers, breakdowns by suite/attack/model, and ablations, please refer to the paper and the released logs/configs.

---

## Repository Structure

A typical layout is as follows (adjust to your actual repo):

```text
.
├── assets/                      # figures for README (1.png, 2.png)
├── agentsentry/                 # core library (diagnostics, purification, enforcement)
├── configs/                     # experiment configs (models, suites, attacks, defenses)
├── scripts/                     # run scripts and helpers
├── evaluation/                  # metrics, aggregation, plotting
└── README.md



Qwen-3-max没有Embeddings,这时候需要切换到pumpkin（或 OpenAI 官方）
<img width="1432" height="974" alt="image" src="https://github.com/user-attachments/assets/42591d7d-52df-4815-972a-608a175aac15" />


然后我们的TCDT方法跑的时候，有时候还会遇到并发key的问题，就是导致流程结束。因为重执行了多次，如下：

<img width="2454" height="1260" alt="image" src="https://github.com/user-attachments/assets/0c483836-9b77-4e0f-891b-e53d36ef81bd" />

<img width="1974" height="936" alt="d8d74b838a954b4bd1bfaf61b6e872e1" src="https://github.com/user-attachments/assets/7b07011a-f292-452b-a473-074a3c2bbc1a" />


因此需要改一下llm的接口，让他不崩溃



<img width="1728" height="710" alt="image" src="https://github.com/user-attachments/assets/850bd5b9-ff30-45d7-a887-4fc7e41479bc" />


执行过程中结果

<img width="1988" height="1311" alt="8d4dda38f6dbaa83bd63f85fef403079" src="https://github.com/user-attachments/assets/0cf7090c-bd6a-45d8-86e1-675956c48d27" />


