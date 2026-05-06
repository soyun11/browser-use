# analyze.py
import json
import argparse
from pathlib import Path
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--dir", default="trajectories")
args = parser.parse_args()

output_dir = Path(args.dir)

total = 0
success = 0
fail = 0
cat_stats = defaultdict(lambda: {"success": 0, "fail": 0})

for f in sorted(output_dir.glob("*.json")):
    with open(f, encoding="utf-8") as fp:
        data = json.load(fp)
    
    total += 1
    category = data.get("category", "unknown")
    
    if data.get("success"):
        success += 1
        cat_stats[category]["success"] += 1
    else:
        fail += 1
        cat_stats[category]["fail"] += 1

print(f"\n{'='*40}")
print(f"폴더: {output_dir}")
print(f"전체: {total}개")
print(f"성공: {success}개")
print(f"실패: {fail}개")
print(f"성공률: {success/total*100:.1f}%" if total > 0 else "데이터 없음")

print(f"\n카테고리별 성공률:")
for cat, stats in cat_stats.items():
    t = stats["success"] + stats["fail"]
    rate = stats["success"] / t * 100 if t > 0 else 0
    print(f"  [{cat}] {stats['success']}/{t} ({rate:.0f}%)")