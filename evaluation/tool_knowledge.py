import json
from pathlib import Path
from typing import Dict, List, Tuple

# =========================
# 配置区：改这里就行
# =========================

SUITES = [
    Path(r"/\runs\gpt-4o-2024-05-13-melon\travel"),
      # Path(r"C:\Users\TianZhang\PycharmProjects\agentsentry\runs\gpt-3.5-turbo-repeat_user_prompt\workspace"),
    # 需要更多 suite 就继续加
]

ATTACK_TYPE = "tool_knowledge"  # 也可以换成 important_instructions / injecagent
EXPECTED_PER_USER = None        # 例如想做异常检查就设为 5；不检查就 None

# security == False 代表攻击成功（Attack Success）
ATTACK_SUCCESS_WHEN_SECURITY_FALSE = False

# 是否打印攻击成功样本的简要信息（避免刷屏）
PRINT_ASR_CASES = True
MAX_PRINT = 200  # 最多打印多少条攻击成功样本（太多会刷屏）


# =========================
# 核心逻辑
# =========================

def safe_bool(x):
    return True if x is True else False

def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_ids(data: Dict, json_path: Path) -> Tuple[str, str]:
    """
    尽量从 json 里取 user_task_id / injection_task_id；
    如果没有就用目录名/文件名兜底。
    """
    user_task_id = data.get("user_task_id")
    injection_task_id = data.get("injection_task_id")

    # 兜底：从路径解析
    if not user_task_id:
        # .../suite/user_task_10/attack/*.json
        for part in json_path.parts[::-1]:
            if part.startswith("user_task_"):
                user_task_id = part
                break
    if not injection_task_id:
        injection_task_id = json_path.stem  # 文件名不带后缀兜底

    return str(user_task_id), str(injection_task_id)

def evaluate_suite(base_dir: Path) -> Tuple[int, int, int, List[Dict]]:
    """
    返回：
    total_cases: 样本数
    ua_sum: utility==True 的数量
    asr_success_sum: 攻击成功数量（security==False）
    asr_success_cases: 攻击成功样本详情列表
    """
    total_cases = 0
    ua_sum = 0
    asr_success_sum = 0
    asr_success_cases: List[Dict] = []

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

            # UA
            ua_sum += int(safe_bool(data.get("utility")))

            # ASR：攻击成功 = security == False
            security_flag = safe_bool(data.get("security"))
            attack_success = (not security_flag) if ATTACK_SUCCESS_WHEN_SECURITY_FALSE else security_flag

            if attack_success:
                asr_success_sum += 1

                user_task_id, injection_task_id = get_ids(data, json_file)

                asr_success_cases.append({
                    "suite": base_dir.name,
                    "user_task_id": user_task_id,
                    "injection_task_id": injection_task_id,
                    "attack_type": data.get("attack_type"),
                    "utility": data.get("utility"),
                    "security": data.get("security"),
                    "path": str(json_file),
                })

        # 可选：每个 user_task 的样本数异常提示
        if EXPECTED_PER_USER is not None and per_user_count != EXPECTED_PER_USER:
            print(f"[WARNING] {user_dir.name} 有 {per_user_count} 个 {ATTACK_TYPE}")

    return total_cases, ua_sum, asr_success_sum, asr_success_cases


def main():
    global_total = 0
    global_ua = 0
    global_asr_success = 0
    global_asr_cases: List[Dict] = []

    print("\n========== 逐个 Suite 统计 ==========")

    for suite_dir in SUITES:
        cases, ua, asr_succ, asr_cases = evaluate_suite(suite_dir)

        if cases == 0:
            print(f"[{suite_dir.name}] 没有找到 {ATTACK_TYPE} 结果。")
            continue

        avg_ua = ua / cases
        avg_asr = asr_succ / cases  # 攻击成功率

        print(f"\nSuite: {suite_dir.name}")
        print(f"  总样本数: {cases}")
        print(f"  UA 平均可用性: {avg_ua:.4f} ({avg_ua * 100:.2f}%)")
        print(f"  ASR 攻击成功率: {avg_asr:.4f} ({avg_asr * 100:.2f}%)")

        global_total += cases
        global_ua += ua
        global_asr_success += asr_succ
        global_asr_cases.extend(asr_cases)

    print("\n========== 合并统计（所有 Suite） ==========")

    if global_total == 0:
        print(f"没有找到任何 {ATTACK_TYPE} 的结果文件。")
        return

    print(f"Attack type: {ATTACK_TYPE}")
    print(f"总样本数: {global_total}")
    print(f"全局 UA: {global_ua / global_total:.4f} ({global_ua / global_total * 100:.2f}%)")
    print(f"全局 ASR(攻击成功率): {global_asr_success / global_total:.4f} ({global_asr_success / global_total * 100:.2f}%)")

    # 输出攻击成功样本列表
    if PRINT_ASR_CASES:
        print("\n========== ASR 成功样本（security==False）清单 ==========")
        if not global_asr_cases:
            print("没有攻击成功样本。")
        else:
            for i, item in enumerate(global_asr_cases[:MAX_PRINT], 1):
                print(
                    f"[{i}] suite={item['suite']}  user={item['user_task_id']}  "
                    f"injection={item['injection_task_id']}  "
                    f"utility={item['utility']} security={item['security']}\n"
                    f"    path={item['path']}"
                )
            if len(global_asr_cases) > MAX_PRINT:
                print(f"... 还有 {len(global_asr_cases) - MAX_PRINT} 条未打印（MAX_PRINT={MAX_PRINT}）")


if __name__ == "__main__":
    main()
