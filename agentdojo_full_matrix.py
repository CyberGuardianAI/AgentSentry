import subprocess
import sys
from datetime import datetime

# =========================
# 实验配置（主实验矩阵）
# =========================

MODELS = [
    "GPT_3_5_TURBO_0125",
    "GPT_4O_2024_05_13",
    "QWEN3_MAX_2025_09_23",
]

ATTACKS = [
    "important_instructions",
    "tool_knowledge",
    "injecagent",
]

SUITES = [
    "travel",
    "workspace",
    "banking",
    "slack",
]

# defense=None 表示 baseline（不加 --defense）
DEFENSES = [
    None,  # baseline: no defense
    "spotlighting_with_delimiting",
    "repeat_user_prompt",
    "tool_filter",
    "transformers_pi_detector",
    "melon",
    "melon_aug",
    "task_shield",
    "tcdt",
]

PYTHON_BIN = sys.executable   # 保证使用当前 python 环境


# =========================
# 执行逻辑
# =========================

def run_command(model: str, attack: str, suite: str, defense) -> bool:
    cmd = [
        PYTHON_BIN,
        "-m", "agentdojo.scripts.benchmark",
        "--model", model,
        "--attack", attack,
        "-s", suite,
    ]

    # baseline 不加 --defense
    if defense is not None:
        cmd += ["--defense", str(defense)]

    defense_name = "none(baseline)" if defense is None else str(defense)

    print("\n" + "=" * 110)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始运行")
    print(f"Model  : {model}")
    print(f"Attack : {attack}")
    print(f"Suite  : {suite}")
    print(f"Defense: {defense_name}")
    print("Command:")
    print(" ".join(cmd))
    print("=" * 110)

    try:
        subprocess.run(cmd, check=True)
        print(f"[✓] 完成: {model} | {attack} | {suite} | {defense_name}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"[✗] 失败: {model} | {attack} | {suite} | {defense_name}")
        print(f"    返回码: {e.returncode}")
        return False


def main():
    print("\n========== AgentDojo 全量实验自动执行 ==========")
    print(f"Models  : {MODELS}")
    print(f"Attacks : {ATTACKS}")
    print(f"Suites  : {SUITES}")
    print(f"Defenses: {[('none(baseline)' if d is None else d) for d in DEFENSES]}")
    print("=" * 110)

    failed = []

    for model in MODELS:
        for attack in ATTACKS:
            for suite in SUITES:
                for defense in DEFENSES:
                    ok = run_command(
                        model=model,
                        attack=attack,
                        suite=suite,
                        defense=defense,
                    )
                    if not ok:
                        failed.append((model, attack, suite, defense))

    print("\n========== 实验全部执行完毕 ==========")

    if failed:
        print("\n[WARNING] 以下实验失败（其余已继续执行）：")
        for model, attack, suite, defense in failed:
            defense_name = "none(baseline)" if defense is None else str(defense)
            print(f"  - model={model}, attack={attack}, suite={suite}, defense={defense_name}")
    else:
        print("[SUCCESS] 所有实验均成功完成")


if __name__ == "__main__":
    main()


