import json
import os
from pathlib import Path

# 你的 workspace 路径
BASE_DIR = Path(os.environ.get("AGENTSENTRY_RUN_ROOT", "runs/qwen3-max-2025-09-23-agentsentry/banking"))

total_cases = 0
utility_sum = 0  # 用来算平均可用性
asr_sum = 0      # 用来算平均 ASR（攻击成功率）

# 遍历所有 user_task_*
for user_dir in BASE_DIR.glob("user_task_*"):
    important_dir = user_dir / "injecagent"
    if not important_dir.is_dir():
        continue

    for json_file in important_dir.glob("*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 双保险：只统计 attack_type == "important_instructions"
        if data.get("attack_type") != "injecagent":
            continue

        total_cases += 1

        # -------- 可用性（UA）--------
        # utility 为 true → 记 1，否则 0
        if data.get("utility") is True:
            utility_sum += 1

        # -------- 攻击成功率（ASR）--------
        # 按 AgentDojo 定义：security 为 False 说明攻击成功 → 记 1
        if data.get("security") is True:
            asr_sum += 1

        # 如果你真想“security 为 False 记为 0，True 记为 1”，
        # 就把上面这段改成：
        # if data.get("security") is True:
        #     asr_sum += 1

# 统计结果
if total_cases == 0:
    print("没有找到任何 injecagent 的结果文件。")
else:
    avg_ua = utility_sum / total_cases
    avg_asr = asr_sum / total_cases

    print(f"总样本数（injecagent）：{total_cases}")
    print(f"平均可用性 UA: {avg_ua:.4f} ({avg_ua * 100:.2f}%)")
    print(f"平均攻击成功率 ASR: {avg_asr:.4f} ({avg_asr * 100:.2f}%)")
