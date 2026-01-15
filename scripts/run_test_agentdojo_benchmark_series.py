import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ======================================================
# 自动实验队列（按顺序执行，跑完一个自动下一个）
# - 模型：QWEN3_MAX_2025_09_23
# - Suite：travel
# - Attack：tool_knowledge
# - 对每个 defense 同时跑两组：
#     (1) under-attack: 传 --attack tool_knowledge
#     (2) clean/no-attack: 不传 --attack
# ======================================================

MODEL = "QWEN3_MAX_2025_09_23"
SUITE = "travel"
ATTACK = "tool_knowledge"

DEFENSES = [
    "melon",
    "tool_filter",
    "spotlighting_with_delimiting",
    "repeat_user_prompt",
    "transformers_pi_detector",
    None,  # baseline：不加 --defense
]

# ======================================================
# 自动生成实验队列：每个 defense 跑 (attack, clean)
# ======================================================

EXPERIMENTS: list[dict] = []
for defense in DEFENSES:
    # 1) Under attack
    EXPERIMENTS.append(
        {
            "model": MODEL,
            "suite": SUITE,
            "attack": ATTACK,
            "defense": defense,
        }
    )
    # 2) Clean / no attack
    EXPERIMENTS.append(
        {
            "model": MODEL,
            "suite": SUITE,
            "attack": None,  # 关键：表示不传 --attack
            "defense": defense,
        }
    )

# ======================================================
# 执行设置
# ======================================================

PYTHON_BIN = sys.executable
LOG_DIR = Path("run_logs_qwen3max")
LOG_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# ======================================================
# 工具函数
# ======================================================

def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _require_env():
    """
    AgentDojo -> openai.OpenAI()（空参）只认 OpenAI SDK 的环境变量：
      - OPENAI_API_KEY (必须)
      - OPENAI_BASE_URL (走 DashScope 时建议设置；未设置则补默认)
    """
    api_key_ok = bool(os.getenv("OPENAI_API_KEY"))
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key_ok:
        raise RuntimeError(
            "Missing environment variable: OPENAI_API_KEY\n"
            "Please set it in this terminal before running, e.g.:\n"
            "  PowerShell: $env:OPENAI_API_KEY='sk-1d467787b9374512a525e3f457760e47'\n"
        )

    if not base_url:
        os.environ["OPENAI_BASE_URL"] = DEFAULT_BASE_URL


def _build_cmd(exp: dict) -> list[str]:
    cmd = [
        PYTHON_BIN,
        "-m", "agentdojo.scripts.benchmark",
        "--model", exp["model"],
        "-s", exp["suite"],
    ]

    # clean/no-attack：不传 --attack
    if exp.get("attack"):
        cmd += ["--attack", exp["attack"]]

    if exp.get("defense"):
        cmd += ["--defense", exp["defense"]]

    return cmd


def run_experiment(idx: int, exp: dict) -> bool:
    cmd = _build_cmd(exp)

    stamp = _now_tag()
    defense_tag = exp.get("defense") or "no_defense"
    attack_tag = exp.get("attack") or "no_attack"
    log_path = LOG_DIR / f"exp{idx:02d}_{exp['suite']}_{attack_tag}_{defense_tag}_{stamp}.log"

    print("\n" + "=" * 100)
    print(f"[Experiment {idx}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Command:")
    print(" ".join(cmd))
    print(f"Log: {log_path}")
    print("=" * 100)

    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"OPENAI_BASE_URL: {os.getenv('OPENAI_BASE_URL')}\n")
            f.write("-" * 100 + "\n\n")
            subprocess.run(cmd, check=True, stdout=f, stderr=f)

        print(f"[✓] Experiment {idx} finished successfully")
        return True

    except subprocess.CalledProcessError as e:
        print(f"[✗] Experiment {idx} FAILED (return code={e.returncode})")
        print(f"    See log for details: {log_path}")
        return False


def main():
    _require_env()

    print("\n========== AgentDojo 自动实验队列 ==========")
    print(f"Total experiments: {len(EXPERIMENTS)}")
    print(f"Python: {PYTHON_BIN}")
    print(f"Model: {MODEL}")
    print(f"Suite: {SUITE}")
    print(f"Attack: {ATTACK} (plus clean/no-attack runs)")
    print(f"OPENAI_BASE_URL: {os.getenv('OPENAI_BASE_URL')}")
    print(f"Logs: {LOG_DIR.resolve()}")

    failed: list[tuple[int, dict]] = []

    for idx, exp in enumerate(EXPERIMENTS, start=1):
        ok = run_experiment(idx, exp)
        if not ok:
            failed.append((idx, exp))

    print("\n========== 实验队列结束 ==========")

    if failed:
        print("\n[WARNING] 以下实验失败：")
        for idx, exp in failed:
            print(
                f"  - Experiment {idx}: "
                f"model={exp['model']} "
                f"suite={exp['suite']} "
                f"attack={exp.get('attack') or 'no_attack'} "
                f"defense={exp.get('defense') or 'no_defense'}"
            )
        print(f"\nLogs directory: {LOG_DIR.resolve()}")
    else:
        print("[SUCCESS] 所有实验均已顺利完成")
        print(f"Logs directory: {LOG_DIR.resolve()}")


if __name__ == "__main__":
    main()
