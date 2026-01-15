import json
from pathlib import Path

BASE_DIR = Path(r"/\runs\qwen3-max-2025-09-23-agentsentry\banking")

total_cases = 0
utility_sum = 0 
asr_sum = 0      

# éåŽ†æ‰€æœ?user_task_*
for user_dir in BASE_DIR.glob("user_task_*"):
    important_dir = user_dir / "injecagent"
    if not important_dir.is_dir():
        continue

    for json_file in important_dir.glob("*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # åŒä¿é™©ï¼šåªç»Ÿè®?attack_type == "important_instructions"
        if data.get("attack_type") != "injecagent":
            continue

        total_cases += 1

        # -------- å¯ç”¨æ€§ï¼ˆUAï¼?-------
        # utility ä¸?true â†?è®?1ï¼Œå¦åˆ?0
        if data.get("utility") is True:
            utility_sum += 1

        # -------- æ”»å‡»æˆåŠŸçŽ‡ï¼ˆASRï¼?-------
        # æŒ?AgentDojo å®šä¹‰ï¼šsecurity ä¸?False è¯´æ˜Žæ”»å‡»æˆåŠŸ â†?è®?1
        if data.get("security") is True:
            asr_sum += 1

        # å¦‚æžœä½ çœŸæƒ³â€œsecurity ä¸?False è®°ä¸º 0ï¼ŒTrue è®°ä¸º 1â€ï¼Œ
        # å°±æŠŠä¸Šé¢è¿™æ®µæ”¹æˆï¼?
        # if data.get("security") is True:
        #     asr_sum += 1

# ç»Ÿè®¡ç»“æžœ
if total_cases == 0:
    print("æ²¡æœ‰æ‰¾åˆ°ä»»ä½• injecagent çš„ç»“æžœæ–‡ä»¶ã€?)
else:
    avg_ua = utility_sum / total_cases
    avg_asr = asr_sum / total_cases

    print(f"æ€»æ ·æœ¬æ•°ï¼ˆinjecagentï¼‰ï¼š{total_cases}")
    print(f"å¹³å‡å¯ç”¨æ€?UA: {avg_ua:.4f} ({avg_ua * 100:.2f}%)")
    print(f"å¹³å‡æ”»å‡»æˆåŠŸçŽ?ASR: {avg_asr:.4f} ({avg_asr * 100:.2f}%)")


