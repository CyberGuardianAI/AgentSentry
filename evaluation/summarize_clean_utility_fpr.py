import os
import json

# 结果根目录
ROOT = os.environ.get("AGENTSENTRY_RUN_ROOT", os.path.join("runs", "gpt-3.5-turbo-agentsentry"))

# 四个 AgentDojo 场景
SUITES = ["workspace", "banking", "slack", "travel"]


def collect_suite_stats(suite_name):
    """
    在某个场景下，遍历所有 user_task_* 目录，
    统计它们的 none（无攻击）结果：
      - 可用性 = utility == True 的比例
      - 误报率 = security == False 的比例（因为任务本身安全）
    额外记录：utility == False 的案例编号，用于排查。
    """
    suite_root = os.path.join(ROOT, suite_name)
    if not os.path.isdir(suite_root):
        print(f"[跳过] 场景 {suite_name} 的目录不存在：{suite_root}")
        return None

    total = 0
    utility_ok = 0
    security_ok = 0
    false_positive = 0
    user_tasks_counted = 0

    # 记录 utility=False 的具体案例
    utility_bad_cases = []  # 每个元素是 dict：{"user_task": ..., "file": ..., "user_task_id": ...}

    for entry in os.listdir(suite_root):
        # 只选 user_task_xx 目录
        if not entry.startswith("user_task_"):
            continue

        user_task_dir = os.path.join(suite_root, entry)
        none_dir = os.path.join(user_task_dir, "none")
        if not os.path.isdir(none_dir):
            # 这个 user_task 没有跑 none，就跳过
            continue

        user_tasks_counted += 1

        for fname in os.listdir(none_dir):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(none_dir, fname)

            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"[警告] 读取 {fpath} 失败：{e}")
                continue

            total += 1
            util = data.get("utility")
            sec = data.get("security")

            # 方便定位：从 json 中再读一次 user_task_id（和路径里的 user_task_x 做个交叉校验）
            user_task_id = data.get("user_task_id", entry)

            if util is True:
                utility_ok += 1
            elif util is False:
                # 记录可用性失败的案例
                utility_bad_cases.append(
                    {
                        "user_task": entry,
                        "file": fname,
                        "user_task_id": user_task_id,
                    }
                )

            if sec is True:
                security_ok += 1
            if sec is False:
                false_positive += 1

    if total == 0:
        print(f"[提示] 场景 {suite_name} 在 none 下没有任何结果。")
        return None

    return {
        "suite": suite_name,
        "user_tasks": user_tasks_counted,
        "total": total,
        "utility_ok": utility_ok,
        "security_ok": security_ok,
        "false_positive": false_positive,
        "utility_rate": utility_ok / total,
        "security_true_rate": security_ok / total,
        "false_positive_rate": false_positive / total,
        "utility_bad_cases": utility_bad_cases,
    }


def main():
    all_stats = []
    for suite in SUITES:
        stats = collect_suite_stats(suite)
        if stats is None:
            continue
        all_stats.append(stats)

        print(f"\n=== 场景：{suite} ===")
        print(f"覆盖的 user_task 数量：{stats['user_tasks']}")
        print(f"样本总数（none 结果文件数）：{stats['total']}")
        print(f"可用性（utility=True 比例）：{stats['utility_rate']:.2%}")
        # 这里 security=False 就是误报（false positive）
        print(
            f"误报率（security=False 比例）：{stats['false_positive_rate']:.2%} "
            f"(FP={stats['false_positive']})"
        )

        # 把 utility=False 的案例编号打印出来
        bad_cases = stats["utility_bad_cases"]
        if bad_cases:
            print(f"可用性为 False 的案例数：{len(bad_cases)}")
            print("具体案例（user_task / user_task_id / 文件名）：")
            for case in bad_cases:
                print(
                    f"  - {case['user_task']} "
                    f"(user_task_id={case['user_task_id']}), file={case['file']}"
                )
        else:
            print("所有 none 结果的 utility 均为 True。")

    # 四个场景整体加权平均
    if all_stats:
        total_all = sum(s["total"] for s in all_stats)
        utility_ok_all = sum(s["utility_ok"] for s in all_stats)
        fp_all = sum(s["false_positive"] for s in all_stats)

        print("\n==== 所有场景总体统计（加权）====")
        print(f"总样本数：{total_all}")
        print(f"整体可用性：{utility_ok_all / total_all:.2%}")
        print(f"整体误报率：{fp_all / total_all:.2%} (FP={fp_all})")
    else:
        print("\n没有任何场景被成功统计。")


if __name__ == "__main__":
    main()
