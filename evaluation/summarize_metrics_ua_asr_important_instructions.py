# -*- coding: utf-8 -*-
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# =========================
# 配置区：改这里即可
# =========================

SUITES = [
    Path(r"C:\Users\TianZhang\PycharmProjects\agentsentry\runs\qwen3-max-2025-09-23-agentsentry\workspace"),
    # 需要更多 suite 就继续加
]

ATTACK_TYPE = "important_instructions"
EXPECTED_PER_USER: Optional[int] = 14  # 你这里是 14 个 injection；不想检查就设 None
PRINT_ATTACK_SUCCESS_CASES = True
MAX_PRINT = 300  # 防止刷屏

# 你的口径：security == True 代表攻击成功（按你要求）
ATTACK_SUCCESS_WHEN_SECURITY_TRUE = True


# =========================
# 工具函数
# =========================

def safe_bool(x) -> bool:
    return True if x is True else False

def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def infer_user_task_id(json_path: Path) -> str:
    # .../workspace/user_task_10/important_instructions/*.json
    for part in reversed(json_path.parts):
        if part.startswith("user_task_"):
            return part
    return "unknown_user_task"

def infer_injection_id(data: Dict, json_path: Path) -> str:
    # 优先用 json 里显式字段，其次用文件名
    return str(data.get("injection_task_id") or data.get("injection_id") or json_path.stem)

def format_case_line(i: int, suite_name: str, user_task_id: str, inj_id: str, utility: bool, security: bool, path: str) -> str:
    return (
        f"[{i}] suite={suite_name}  user={user_task_id}  injection={inj_id}  "
        f"utility={utility}  security={security}\n"
        f"    path={path}"
    )


# =========================
# 核心逻辑
# =========================

def evaluate_suite(base_dir: Path) -> Tuple[int, int, int, List[Dict]]:
    """
    返回：
    total_cases: 样本数
    ua_sum: utility==True 数量
    asr_sum: 攻击成功数量（按 security==True 的口径）
    asr_cases: 攻击成功样本详情列表
    """
    total_cases = 0
    ua_sum = 0
    asr_sum = 0
    asr_cases: List[Dict] = []

    print(f"\n=== 检查 suite: {base_dir.name} ===")

    for user_dir in base_dir.glob("user_task_*"):
        atk_dir = user_dir / ATTACK_TYPE
        if not atk_dir.is_dir():
            continue

        per_user_count = 0

        for json_file in atk_dir.glob("*.json"):
            try:
                data = load_json(json_file)
            except Exception as e:
                print(f"[ERROR] 读取失败: {json_file} ({e})")
                continue

            if data.get("attack_type") != ATTACK_TYPE:
                continue

            per_user_count += 1
            total_cases += 1

            utility_flag = safe_bool(data.get("utility"))
            security_flag = safe_bool(data.get("security"))

            ua_sum += int(utility_flag)

            # 攻击成功口径：security==True
            attack_success = security_flag if ATTACK_SUCCESS_WHEN_SECURITY_TRUE else (not security_flag)
            if attack_success:
                asr_sum += 1

                user_task_id = str(data.get("user_task_id") or infer_user_task_id(json_file))
                injection_id = infer_injection_id(data, json_file)

                asr_cases.append({
                    "suite": base_dir.name,
                    "user_task_id": user_task_id,
                    "injection_task_id": injection_id,
                    "utility": utility_flag,
                    "security": security_flag,
                    "path": str(json_file),
                })

        # 每个 user_task 的样本数异常提示
        if EXPECTED_PER_USER is not None and per_user_count != EXPECTED_PER_USER:
            print(f"[WARNING] {user_dir.name} 有 {per_user_count} 个 {ATTACK_TYPE}（期望 {EXPECTED_PER_USER}）")

    return total_cases, ua_sum, asr_sum, asr_cases


def main():
    global_total = 0
    global_ua = 0
    global_asr = 0
    global_asr_cases: List[Dict] = []

    print("\n========== 逐个 Suite 统计 ==========")

    for suite_dir in SUITES:
        cases, ua, asr, asr_cases = evaluate_suite(suite_dir)

        if cases == 0:
            print(f"[{suite_dir.name}] 没有找到 {ATTACK_TYPE} 结果。")
            continue

        avg_ua = ua / cases
        avg_asr = asr / cases

        print(f"\nSuite: {suite_dir.name}")
        print(f"  总样本数: {cases}")
        print(f"  UA 平均可用性: {avg_ua:.4f} ({avg_ua * 100:.2f}%)")
        print(f"  ASR 平均攻击成功率: {avg_asr:.4f} ({avg_asr * 100:.2f}%)")

        global_total += cases
        global_ua += ua
        global_asr += asr
        global_asr_cases.extend(asr_cases)

    print("\n========== 合并统计（所有 Suite） ==========")

    if global_total == 0:
        print(f"没有找到任何 {ATTACK_TYPE} 的结果文件。")
        return

    print(f"Attack type: {ATTACK_TYPE}")
    print(f"总样本数: {global_total}")
    print(f"全局 UA: {global_ua / global_total:.4f} ({global_ua / global_total * 100:.2f}%)")
    print(f"全局 ASR(攻击成功率): {global_asr / global_total:.4f} ({global_asr / global_total * 100:.2f}%)")

    # 输出攻击成功样本列表（security==True）
    if PRINT_ATTACK_SUCCESS_CASES:
        print("\n========== 攻击成功样本清单（security==True） ==========")
        if not global_asr_cases:
            print("没有攻击成功样本。")
        else:
            for i, item in enumerate(global_asr_cases[:MAX_PRINT], 1):
                print(
                    format_case_line(
                        i,
                        item["suite"],
                        item["user_task_id"],
                        item["injection_task_id"],
                        item["utility"],
                        item["security"],
                        item["path"],
                    )
                )
            if len(global_asr_cases) > MAX_PRINT:
                print(f"... 还有 {len(global_asr_cases) - MAX_PRINT} 条未打印（MAX_PRINT={MAX_PRINT}）")


if __name__ == "__main__":
    main()


