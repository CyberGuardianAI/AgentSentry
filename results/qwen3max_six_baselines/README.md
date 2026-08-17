# Qwen3-Max Results for Six Additional Baselines

This focused release reports the completed Qwen3-Max comparison for CausalArmor, AttriGuard, IPIGuard, AgentArmor, TraceAegis, and ProvSEEK-adapted on AgentDojo. All values below are percentages.

| Defense | CU | FPR | Important Instr. UA / ASR | Tool Knowledge UA / ASR | InjecAgent UA / ASR | Macro UA / ASR |
|---|---:|---:|---:|---:|---:|---:|
| CausalArmor | 69.07 | 65.98 | 49.62 / 13.23 | 33.57 / 21.43 | 77.08 / 1.39 | 53.43 / 12.02 |
| AttriGuard | 76.29 | 23.71 | 79.70 / 4.66 | 68.57 / 16.43 | 76.39 / 0.69 | 74.89 / 7.26 |
| IPIGuard | 77.32 | 39.18 | 64.36 / 4.96 | 68.57 / 12.86 | 68.75 / 4.86 | 67.23 / 7.56 |
| AgentArmor | 77.32 | >=9.28 | 60.30 / 0.00 | 26.43 / 8.57 | 63.19 / 0.00 | 49.97 / 2.86 |
| TraceAegis | 65.98 | >=34.02 | 54.29 / 3.46 | 27.14 / 5.00 | 61.11 / 0.00 | 47.51 / 2.82 |
| ProvSEEK-adapted | 53.61 | >=50.52 | 15.64 / 4.51 | 29.29 / 45.71 | 45.14 / 0.69 | 30.02 / 16.97 |

## Evaluation Scope

- Benign tasks: 97 cases for CU and FPR.
- Important Instructions: 665 attack cases.
- Tool Knowledge: 140 attack cases.
- InjecAgent: 144 attack cases.
- Macro UA/ASR: unweighted mean over the three attack families.
- Exact uncertainty: two-sided 95% Clopper--Pearson intervals are included in `qwen3max_six_baselines.json` and `qwen3max_six_baselines.csv`.

CU and UA are higher-is-better; FPR and ASR are lower-is-better. Counts, denominators, intervals, pooled summaries, and evidence labels are preserved in the machine-readable files.

## Evidence Notes

- CausalArmor uses an independent public implementation because official author code was not verified. Its attack counts are integer reconstructions from the audited aggregate table.
- AttriGuard uses the official Zenodo artifact.
- IPIGuard uses the official code and bundled AgentDojo fork.
- AgentArmor and TraceAegis are independent paper-faithful AgentDojo reproductions because author method code was unavailable.
- ProvSEEK-adapted is an online AgentDojo adaptation of the official provenance architecture and is not the original ProvSEEK evaluation metric.
- AgentArmor, TraceAegis, and ProvSEEK-adapted FPR entries are lower bounds from explicit block markers; their true intervention FPR may be higher.

This directory contains the requested aggregate result release only; raw trajectories and unrelated experiment outputs are not included.
