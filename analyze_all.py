# analyze_all.py
# 모든 모델의 도서관/증명서 성공률을 한 번에 비교 출력

import json
from pathlib import Path
from collections import defaultdict

MODELS = [
    ("ChatBrowserUse",      "trajectories",         "cert_trajectories"),
    ("GPT-4o-mini",         "trajectories_gpt",     "cert_trajectories_gpt"),
    ("Gemini-2.5-flash",    "trajectories_gemini",  "cert_trajectories_gemini"),
    ("Qwen2.5-32B (FT)",    "trajectories_finetuned", "cert_trajectories_finetuned"),
    ("Qwen2.5-VL-32B (FT)", "trajectories_finetuned_vl", "cert_trajectories_finetuned_vl"),
]


def analyze_dir(folder: str) -> dict:
    path = Path(folder)
    if not path.exists():
        return {"total": 0, "success": 0, "rate": None, "categories": {}}

    results = {"total": 0, "success": 0, "categories": defaultdict(lambda: {"success": 0, "total": 0})}

    for f in path.glob("*.json"):
        try:
            data = json.load(open(f, encoding="utf-8"))
            results["total"] += 1
            cat = data.get("category", "unknown")
            results["categories"][cat]["total"] += 1
            if data.get("success"):
                results["success"] += 1
                results["categories"][cat]["success"] += 1
        except Exception:
            continue

    results["rate"] = results["success"] / results["total"] * 100 if results["total"] > 0 else 0
    return results


def print_summary():
    print("\n" + "="*70)
    print("📊 CNU AI Agent - 모델별 성공률 비교")
    print("="*70)

    # 전체 요약 테이블
    print(f"\n{'모델':<25} {'도서관':>12} {'증명서':>12}")
    print("-"*50)

    all_results = {}
    for model_name, lib_dir, cert_dir in MODELS:
        lib = analyze_dir(lib_dir)
        cert = analyze_dir(cert_dir)
        all_results[model_name] = {"library": lib, "cert": cert}

        lib_str = f"{lib['success']}/{lib['total']} ({lib['rate']:.1f}%)" if lib['total'] > 0 else "미수집"
        cert_str = f"{cert['success']}/{cert['total']} ({cert['rate']:.1f}%)" if cert['total'] > 0 else "미수집"
        print(f"{model_name:<25} {lib_str:>12} {cert_str:>12}")

    # 카테고리별 상세 출력
    for model_name, lib_dir, cert_dir in MODELS:
        lib = all_results[model_name]["library"]
        cert = all_results[model_name]["cert"]

        if lib["total"] == 0 and cert["total"] == 0:
            continue

        print(f"\n{'='*70}")
        print(f"🤖 {model_name}")
        print('='*70)

        if lib["total"] > 0:
            print(f"\n  📚 도서관 ({lib['success']}/{lib['total']}, {lib['rate']:.1f}%)")
            for cat, stats in sorted(lib["categories"].items()):
                rate = stats["success"] / stats["total"] * 100 if stats["total"] > 0 else 0
                print(f"    [{cat}] {stats['success']}/{stats['total']} ({rate:.0f}%)")

        if cert["total"] > 0:
            print(f"\n  📋 증명서 ({cert['success']}/{cert['total']}, {cert['rate']:.1f}%)")
            for cat, stats in sorted(cert["categories"].items()):
                rate = stats["success"] / stats["total"] * 100 if stats["total"] > 0 else 0
                print(f"    [{cat}] {stats['success']}/{stats['total']} ({rate:.0f}%)")

    print("\n" + "="*70)


if __name__ == "__main__":
    print_summary()