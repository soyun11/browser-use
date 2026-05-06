# preprocess.py
# trajectories 폴더들에서 success:true인 것만 골라 파인튜닝 포맷으로 변환
# 사용법:
#   python preprocess.py --model bu      # ChatBrowserUse만
#   python preprocess.py --model gpt     # GPT-4o-mini만
#   python preprocess.py --model gemini  # Gemini만
#   python preprocess.py --model all     # 전체 합치기

import json
import argparse
from pathlib import Path

# ──────────────────────────────────────────
# 모델별 trajectory 폴더 매핑
# ──────────────────────────────────────────
MODEL_DIRS = {
    "bu": [
        "./trajectories",
        "./cert_trajectories",
    ],
    "gpt": [
        "./trajectories_gpt",
        "./cert_trajectories_gpt",
    ],
    "gemini": [
        "./trajectories_gemini",
        "./cert_trajectories_gemini",
    ],
}


# ──────────────────────────────────────────
# trajectory → 파인튜닝 messages 변환
# ──────────────────────────────────────────
def extract_messages(traj: dict) -> list[dict]:
    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI agent that automates tasks on Chungnam National University web systems. "
                "Given a task, you navigate websites, interact with elements, and complete the task step by step."
            )
        },
        {
            "role": "user",
            "content": traj["task"]
        }
    ]

    for step in traj["steps"]:
        actions = step.get("actions", [])
        if not actions:
            continue

        # assistant: 액션 출력
        action_content = {
            "memory": step.get("memory", "") or "",
            "next_goal": step.get("next_goal", ""),
            "actions": actions,
        }
        messages.append({
            "role": "assistant",
            "content": json.dumps(action_content, ensure_ascii=False)
        })

        # user: 액션 결과 피드백
        results = step.get("results", [])
        result_memories = [r["memory"] for r in results if r.get("memory")]
        errors = [r["error"] for r in results if r.get("error")]
        is_done = any(r.get("is_done") for r in results)

        feedback_parts = []
        if result_memories:
            feedback_parts.append("\n".join(result_memories))
        if errors:
            feedback_parts.append("Error: " + "\n".join(errors))
        if is_done:
            feedback_parts.append("Task completed.")

        if feedback_parts:
            messages.append({
                "role": "user",
                "content": "\n".join(feedback_parts)
            })

    return messages


# ──────────────────────────────────────────
# 전처리 실행
# ──────────────────────────────────────────
def preprocess(traj_dirs: list[str], output_path: str) -> dict:
    results = {"total": 0, "success": 0, "failed_skipped": 0, "written": 0}
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as fout:
        for traj_dir in traj_dirs:
            traj_path = Path(traj_dir)
            if not traj_path.exists():
                print(f"⚠️  폴더 없음: {traj_dir}")
                continue

            json_files = sorted(traj_path.glob("*.json"))
            print(f"\n📂 {traj_dir}: {len(json_files)}개 파일")

            for f in json_files:
                results["total"] += 1
                try:
                    with open(f, encoding="utf-8") as fin:
                        traj = json.load(fin)

                    if not traj.get("success"):
                        results["failed_skipped"] += 1
                        continue

                    results["success"] += 1
                    messages = extract_messages(traj)

                    if len(messages) < 3:
                        continue

                    fout.write(json.dumps({
                        "messages": messages,
                        "metadata": {
                            "task": traj["task"],
                            "category": traj.get("category", ""),
                            "total_steps": traj.get("total_steps", 0),
                            "total_duration_seconds": traj.get("total_duration_seconds", 0),
                            "source_file": f.name,
                            "source_dir": str(traj_dir),
                        }
                    }, ensure_ascii=False) + "\n")
                    results["written"] += 1

                except Exception as e:
                    print(f"  ❌ {f.name} 처리 실패: {e}")

    print(f"\n{'='*50}")
    print(f"전체: {results['total']}개")
    print(f"성공 (사용): {results['success']}개")
    print(f"실패 (스킵): {results['failed_skipped']}개")
    print(f"최종 저장: {results['written']}개 → {output_path}")
    return results


# ──────────────────────────────────────────
# 메인
# ──────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["bu", "gpt", "gemini", "all"], default="all")
    args = parser.parse_args()

    if args.model == "all":
        # 전체 합치기
        all_dirs = []
        for dirs in MODEL_DIRS.values():
            all_dirs.extend(dirs)
        preprocess(all_dirs, "./data/ft_train_all.jsonl")

    else:
        # 모델별 개별 처리
        dirs = MODEL_DIRS[args.model]
        preprocess(dirs, f"./data/ft_train_{args.model}.jsonl")