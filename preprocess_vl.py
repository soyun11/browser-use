# preprocess_vl.py
# trajectories_vl/ 폴더에서 success:true인 것만 골라
# 이미지 포함 + is_meaningful 필터링 + 축약 시스템 프롬프트 적용
# 사용법:
#   python preprocess_vl.py --model bu
#   python preprocess_vl.py --model gpt
#   python preprocess_vl.py --model gemini
#   python preprocess_vl.py --model all

import json
import re
import argparse
from pathlib import Path

# ──────────────────────────────────────────
# 모델별 trajectory 폴더 매핑 (VL 버전)
# ──────────────────────────────────────────
MODEL_DIRS = {
    "bu": [
        "./trajectories_vl",
        "./cert_trajectories_vl",
        "./cert_trajectories_vl_gemini",
    ],
    "gpt": [
        "./trajectories_vl_gpt",
        "./cert_trajectories_vl_gpt",
    ],
    "gemini": [
        "./trajectories_vl_gemini",
        "./cert_trajectories_vl_gemini",
    ],
}

# training_data_vl.jsonl 경로 (이미지 포함 raw 데이터)
TRAINING_DATA_DIRS = {
    "bu": "./data/training_data_vl.jsonl",
    "gpt": "./data/training_data_vl_gpt.jsonl",
    "gemini": "./data/training_data_vl_gemini.jsonl",
}

# ──────────────────────────────────────────
# 축약 시스템 프롬프트
# ──────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an AI agent that automates CNU university web tasks.\n"
    "Input includes:\n"
    "- <agent_history>: Previous actions and results\n"
    "- <browser_state>: Current page elements as [index]<type>text</type>\n"
    "- <user_request>: Task to complete\n\n"
    "Rules:\n"
    "- Only interact with elements that have [index]\n"
    "- Use <secret>placeholder</secret> for credentials\n"
    "- Call done when task is complete or impossible\n"
    "- Set success=true only if fully completed\n\n"
    "Output JSON format:\n"
    '{"thinking": "", "evaluation_previous_goal": "", "memory": "", "next_goal": "", "action": []}'
)

# BASE_CONTEXT 접두사 목록 (task 키 매칭용)
BASE_CONTEXT_PREFIXES = [
    "충남대 도서관 https://library.cnu.ac.kr 에서 다음을 수행해줘: ",
    "충남대 증명서 발급 사이트 https://cnu.icerti.com/icerti/index_internet.jsp?t=3415 에서 다음을 수행해줘. 단, 최종 결제 및 발급 확정 버튼은 절대 누르지 말고 직전 단계까지만 수행해줘: ",
]


# ──────────────────────────────────────────
# trajectory JSON → VL 파인튜닝 messages 변환
# ──────────────────────────────────────────
def extract_messages_vl(traj: dict, image_map: dict) -> list[dict]:
    """
    trajectory JSON + image_map → VL 파인튜닝용 messages 리스트
    축약 시스템 프롬프트 + is_meaningful 필터링된 이미지 사용
    """
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": traj["task"]
        }
    ]

    task_images = image_map.get(traj["task"], [])
    image_idx = 0  # 이미지 인덱스 (meaningful 스텝만 카운트)

    for step_idx, step in enumerate(traj["steps"]):
        actions = step.get("actions", [])
        if not actions:
            continue

        # # assistant: 이 스텝에서 취한 액션
        # action_content = {
        #     "memory": step.get("memory", "") or "",
        #     "next_goal": step.get("next_goal", ""),
        #     "actions": actions,
        # }
        # messages.append({
        #     "role": "assistant",
        #     "content": json.dumps(action_content, ensure_ascii=False)
        # })
        # ── done 액션 success=True 강제 변환 ──
        fixed_actions = []
        for action in actions:
            if action.get("type") == "done":
                action = dict(action)
                params = dict(action.get("params", {}))
                params["success"] = True  # 강제 True
                action["params"] = params
            fixed_actions.append(action)
 
        action_content = {
            "memory": step.get("memory", "") or "",
            "next_goal": step.get("next_goal", ""),
            "actions": fixed_actions,
        }
        messages.append({
            "role": "assistant",
            "content": json.dumps(action_content, ensure_ascii=False)
        })

        # user: 액션 결과 피드백 + 스크린샷
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

        # 이미지가 있으면 멀티모달 content로 구성
        if image_idx < len(task_images) and task_images[image_idx]:
            user_content = []
            if feedback_parts:
                user_content.append({
                    "type": "text",
                    "text": "\n".join(feedback_parts)
                })
            user_content.append({
                "type": "image_url",
                "image_url": {"url": task_images[image_idx]}
            })
            messages.append({
                "role": "user",
                "content": user_content
            })
            image_idx += 1
        elif feedback_parts:
            messages.append({
                "role": "user",
                "content": "\n".join(feedback_parts)
            })

    return messages


# ──────────────────────────────────────────
# training_data_vl.jsonl에서 이미지 맵 생성
# is_meaningful=False 스텝 이미지는 제외
# {task: [meaningful 스텝별 이미지 url 리스트]}
# ──────────────────────────────────────────
def build_image_map(training_data_path: str) -> dict:
    image_map = {}
    path = Path(training_data_path)
    if not path.exists():
        print(f"⚠️  training_data 없음: {training_data_path}")
        return image_map

    total_lines = 0
    meaningful_images = 0
    skipped_images = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                total_lines += 1

                # is_meaningful 필터링 (없으면 True로 간주 - 이전 버전 호환)
                is_meaningful = data.get("is_meaningful", True)
                if is_meaningful is False:
                    skipped_images += 1
                    continue

                task = None
                step_images = []

                for msg in data.get("input", []):
                    if msg.get("role") == "user":
                        content = msg.get("content", "")
                        if isinstance(content, list):
                            for part in content:
                                if isinstance(part, dict):
                                    if part.get("type") == "text":
                                        text = part.get("text", "")
                                        # <user_request> 태그에서 task 추출
                                        if task is None and "<user_request>" in text:
                                            match = re.search(
                                                r"<user_request>(.*?)</user_request>",
                                                text, re.DOTALL
                                            )
                                            if match:
                                                task = match.group(1).strip()
                                    elif part.get("type") == "image_url":
                                        img = part.get("image_url", {}).get("url", "")
                                        if img:
                                            step_images.append(img)

                if task and step_images:
                    meaningful_images += len(step_images)

                    # BASE_CONTEXT 제거한 짧은 버전도 키로 저장
                    short_task = task
                    for prefix in BASE_CONTEXT_PREFIXES:
                        if task.startswith(prefix):
                            short_task = task[len(prefix):]
                            break

                    for key in [task, short_task]:
                        if key not in image_map:
                            image_map[key] = []
                        image_map[key].extend(step_images)

            except Exception:
                continue

    print(f"이미지 맵 생성: {len(image_map)}개 태스크")
    print(f"  meaningful 이미지: {meaningful_images}개 / 스킵: {skipped_images}개 (전체 {total_lines}개 스텝)")
    return image_map


# ──────────────────────────────────────────
# 전처리 실행
# ──────────────────────────────────────────
def preprocess_vl(traj_dirs: list[str], training_data_path: str, output_path: str) -> dict:
    results = {"total": 0, "success": 0, "failed_skipped": 0, "written": 0}
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # 이미지 맵 로드
    image_map = build_image_map(training_data_path)

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
                    messages = extract_messages_vl(traj, image_map)

                    if len(messages) < 3:
                        continue

                    fout.write(json.dumps({
                        "messages": messages,
                        "metadata": {
                            "task": traj["task"],
                            "category": traj.get("category", ""),
                            "total_steps": traj.get("total_steps", 0),
                            "source_file": f.name,
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
    parser.add_argument("--model", choices=["bu", "gpt", "gemini", "all"], default="bu")
    args = parser.parse_args()

    if args.model == "all":
        all_dirs = []
        for dirs in MODEL_DIRS.values():
            all_dirs.extend(dirs)
        preprocess_vl(all_dirs, TRAINING_DATA_DIRS["bu"], "./data/ft_train_vl_all.jsonl")
    else:
        preprocess_vl(
            MODEL_DIRS[args.model],
            TRAINING_DATA_DIRS[args.model],
            f"./data/ft_train_vl_{args.model}_v2.jsonl"
        )