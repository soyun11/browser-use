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
    "gemini_library": [
        "./trajectories_vl_gemini",
    ],
    "gemini_cert": [
        "./cert_trajectories_vl_gemini",
    ],
}

# training_data_vl.jsonl 경로 (이미지 포함 raw 데이터)
TRAINING_DATA_DIRS = {
    "bu": "./data/training_data_vl.jsonl",
    "gpt": "./data/training_data_vl_gpt.jsonl",
    "gemini": "./data/training_data_vl_gemini.jsonl",
    "gemini_library": "./data/training_data_vl_gemini_library.jsonl",
    "gemini_cert": "./data/training_data_vl_gemini_cert.jsonl",
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
# 단독 wait는 유지, 연속 3회 이상 wait만 제외
# {task: [meaningful 스텝별 이미지 url 리스트]}
# ──────────────────────────────────────────
def _is_wait_only_step(data: dict) -> bool:
    """raw step 데이터에서 wait-only 액션인지 확인"""
    output = data.get("output", {})
    action = output.get("action", [])
    if not action:
        return False
    action_types = []
    for a in action:
        if isinstance(a, dict):
            for k, v in a.items():
                if v is not None:
                    action_types.append(k)
                    break
    return action_types == ["wait"]


def _compute_meaningful_flags(steps: list[dict]) -> list[bool]:
    """
    task 내 스텝 시퀀스에서 새로운 meaningful 판정:
    - 원래 meaningful=True → 그대로 유지
    - wait-only 스텝이 연속 3회 미만 → meaningful=True로 복원
    - wait-only 스텝이 연속 3회 이상 → meaningful=False 유지
    - wait 외 이유로 not meaningful → meaningful=False 유지
    """
    n = len(steps)
    result = []
    i = 0
    while i < n:
        step = steps[i]
        if step.get("is_meaningful", True):
            result.append(True)
            i += 1
        elif _is_wait_only_step(step):
            # 연속 wait 구간 끝 탐색
            j = i
            while j < n and not steps[j].get("is_meaningful", True) and _is_wait_only_step(steps[j]):
                j += 1
            consecutive = j - i
            # 3회 미만 → 단독/짧은 wait이므로 meaningful로 복원
            flag = consecutive < 3
            result.extend([flag] * consecutive)
            i = j
        else:
            # wait 외 이유(반복 액션 등)로 not meaningful → 그대로
            result.append(False)
            i += 1
    return result


def build_image_map(training_data_path: str) -> dict:
    from collections import defaultdict

    image_map = {}
    path = Path(training_data_path)
    if not path.exists():
        print(f"⚠️  training_data 없음: {training_data_path}")
        return image_map

    # 1단계: 전체 로드 및 task_index 기준 그룹핑
    all_steps: list[dict] = []
    task_groups: dict[int, list[dict]] = defaultdict(list)

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                all_steps.append(data)
                task_idx = data.get("task_index", 0)
                task_groups[task_idx].append(data)
            except Exception:
                continue

    for task_idx in task_groups:
        task_groups[task_idx].sort(key=lambda s: s.get("step_number", 0))

    # 2단계: task별 연속 wait 판정 → step별 new_meaningful 맵
    step_meaningful: dict[tuple[int, int], bool] = {}  # (task_idx, step_number) → bool
    for task_idx, steps in task_groups.items():
        flags = _compute_meaningful_flags(steps)
        for step, flag in zip(steps, flags):
            key = (task_idx, step.get("step_number", 0))
            step_meaningful[key] = flag

    # 3단계: 이미지 맵 구성
    total_lines = len(all_steps)
    meaningful_images = 0
    skipped_images = 0

    def _extract_task_and_images(data: dict) -> tuple[str | None, list[str]]:
        task = None
        imgs: list[str] = []
        for msg in data.get("input", []):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, list):
                    for part in content:
                        if not isinstance(part, dict):
                            continue
                        if part.get("type") == "text":
                            text = part.get("text", "")
                            if task is None and "<user_request>" in text:
                                m = re.search(r"<user_request>(.*?)</user_request>", text, re.DOTALL)
                                if m:
                                    task = m.group(1).strip()
                        elif part.get("type") == "image_url":
                            img = part.get("image_url", {}).get("url", "")
                            if img:
                                imgs.append(img)
        return task, imgs

    for data in all_steps:
        task_idx = data.get("task_index", 0)
        step_num = data.get("step_number", 0)
        is_meaningful = step_meaningful.get((task_idx, step_num), data.get("is_meaningful", True))

        task, step_images = _extract_task_and_images(data)

        if not is_meaningful:
            skipped_images += len(step_images)
            continue

        if task and step_images:
            meaningful_images += len(step_images)

            short_task = task
            for prefix in BASE_CONTEXT_PREFIXES:
                if task.startswith(prefix):
                    short_task = task[len(prefix):]
                    break

            for key in [task, short_task]:
                if key not in image_map:
                    image_map[key] = []
                image_map[key].extend(step_images)

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
# training_data_vl.jsonl에서 직접 파인튜닝 데이터 생성
# trajectory 폴더 없이도 동작
# ──────────────────────────────────────────
def preprocess_from_jsonl(training_data_path: str, output_path: str) -> dict:
	from collections import defaultdict

	results = {"total_steps": 0, "total_trajectories": 0, "written": 0}
	output = Path(output_path)
	output.parent.mkdir(parents=True, exist_ok=True)

	path = Path(training_data_path)
	if not path.exists():
		print(f"⚠️  파일 없음: {training_data_path}")
		return results

	# ── 헬퍼 함수들 ──

	def _get_task(d: dict) -> str | None:
		for msg in d["input"]:
			if msg["role"] == "user":
				content = msg["content"]
				if isinstance(content, list):
					for p in content:
						if p.get("type") == "text":
							m = re.search(r"<user_request>(.*?)</user_request>", p["text"], re.DOTALL)
							if m:
								return m.group(1).strip()
				elif isinstance(content, str):
					m = re.search(r"<user_request>(.*?)</user_request>", content, re.DOTALL)
					if m:
						return m.group(1).strip()
		return None

	def _get_agent_history(d: dict) -> str:
		for msg in d["input"]:
			if msg["role"] == "user":
				content = msg["content"]
				text = ""
				if isinstance(content, list):
					for p in content:
						if p.get("type") == "text":
							text = p["text"]
							break
				else:
					text = content
				m = re.search(r"<agent_history>(.*?)</agent_history>", text, re.DOTALL)
				if m:
					return m.group(1).strip()
		return ""

	def _get_image(d: dict) -> str | None:
		for msg in d["input"]:
			if msg["role"] == "user":
				content = msg["content"]
				if isinstance(content, list):
					for p in content:
						if p.get("type") == "image_url":
							return p["image_url"]["url"]
		return None

	def _is_wait_only(d: dict) -> bool:
		actions = d["output"].get("action", [])
		if not actions:
			return False
		return all(isinstance(a, dict) and "wait" in a for a in actions)

	def _compute_meaningful_flags(steps: list[dict]) -> list[bool]:
		n = len(steps)
		flags: list[bool] = []
		i = 0
		while i < n:
			if not _is_wait_only(steps[i]):
				flags.append(True)
				i += 1
			else:
				j = i
				while j < n and _is_wait_only(steps[j]):
					j += 1
				consecutive = j - i
				flag = consecutive < 3
				flags.extend([flag] * consecutive)
				i = j
		return flags

	def _extract_result_text(curr_step: dict, prev_step: dict) -> str:
		curr_h = _get_agent_history(curr_step)
		prev_h = _get_agent_history(prev_step)
		delta = curr_h[len(prev_h):].strip()
		m = re.search(r"Result\n(.*?)$", delta, re.DOTALL)
		if m:
			return m.group(1).strip()
		return delta.strip()

	def _split_trajectories(steps: list[dict]) -> list[list[dict]]:
		trajectories: list[list[dict]] = []
		current: list[dict] = []
		for step in steps:
			hist = _get_agent_history(step)
			if hist == "Agent initialized" and current:
				trajectories.append(current)
				current = []
			current.append(step)
		if current:
			trajectories.append(current)
		return trajectories

	# 학습에서 제외할 액션 타입
	SKIP_ACTION_TYPES = {"replace_file", "read_file", "evaluate", "write_file"}

	def _should_skip_step(actions: list) -> bool:
		if not actions:
			return False
		def get_action_type(a):
			if isinstance(a, dict):
				if "type" in a:
					return a["type"]
				return list(a.keys())[0] if a else None
			return None
		return all(
			get_action_type(a) in SKIP_ACTION_TYPES
			for a in actions
		)

	def _build_conversation(traj_steps: list[dict], task: str) -> list[dict]:
		meaningful_flags = _compute_meaningful_flags(traj_steps)
		messages: list[dict] = [
			{"role": "system", "content": SYSTEM_PROMPT},
			{"role": "user", "content": task},
		]
		for i, step in enumerate(traj_steps):
			# assistant: done action의 success 강제 True
			actions = step["output"].get("action", [])

			# replace_file, read_file, evaluate, write_file 액션만 제거
			SKIP = {"replace_file", "read_file", "evaluate", "write_file", "extract"}
			def _get_type(a):
				if "type" in a: return a["type"]
				return list(a.keys())[0] if a else None
			actions = [a for a in actions if _get_type(a) not in SKIP]
			if not actions:
				continue

			fixed_actions = []
			for a in actions:
				if isinstance(a, dict) and "done" in a:
					a = dict(a)
					done_params = dict(a["done"])
					done_params["success"] = True
					a["done"] = done_params
				fixed_actions.append(a)

			action_content = {
				"memory": step["output"].get("memory", "") or "",
				"next_goal": step["output"].get("next_goal", ""),
				"actions": fixed_actions,
			}
			messages.append({"role": "assistant", "content": json.dumps(action_content, ensure_ascii=False)})

			# user: 다음 스텝의 history delta에서 result 추출 + 이미지
			if i + 1 < len(traj_steps):
				next_step = traj_steps[i + 1]
				result_text = _extract_result_text(next_step, step)
				next_img = _get_image(next_step) if meaningful_flags[i + 1] else None

				if next_img:
					user_content: list[dict] | str = []
					if result_text:
						user_content.append({"type": "text", "text": result_text})
					user_content.append({"type": "image_url", "image_url": {"url": next_img}})
					messages.append({"role": "user", "content": user_content})
				elif result_text:
					messages.append({"role": "user", "content": result_text})

		return messages

	# ── 메인 처리 ──
	all_steps: list[dict] = []
	with open(path, encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			try:
				all_steps.append(json.loads(line))
			except Exception:
				continue
	results["total_steps"] = len(all_steps)

	# task별 그룹핑 (원래 순서 유지)
	task_steps: dict[str, list[dict]] = defaultdict(list)
	for d in all_steps:
		task = _get_task(d)
		if task:
			task_steps[task].append(d)

	skipped_wait = 0
	with open(output, "w", encoding="utf-8") as fout:
		for task, steps in task_steps.items():
			trajectories = _split_trajectories(steps)
			for traj in trajectories:
				results["total_trajectories"] += 1
				flags = _compute_meaningful_flags(traj)
				skipped_wait += sum(1 for f in flags if not f)
				messages = _build_conversation(traj, task)
				if len(messages) < 3:
					continue
				fout.write(json.dumps({
					"messages": messages,
					"metadata": {
						"task": task,
						"total_steps": len(traj),
					}
				}, ensure_ascii=False) + "\n")
				results["written"] += 1

	print(f"\n{'='*50}")
	print(f"전체 스텝: {results['total_steps']}개")
	print(f"총 태스크: {len(task_steps)}개")
	print(f"총 trajectory: {results['total_trajectories']}개")
	print(f"연속 wait 스킵 스텝: {skipped_wait}개")
	print(f"최종 저장: {results['written']}개 → {output_path}")
	return results


# ──────────────────────────────────────────
# 메인
# ──────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["bu", "gpt", "gemini", "gemini_library", "gemini_cert", "all"], default="bu")
    parser.add_argument("--output", type=str, default=None, help="출력 파일 경로 (기본값: ./data/ft_train_vl_{model}_v2.jsonl)")
    parser.add_argument("--from-jsonl", action="store_true", help="trajectory 폴더 대신 training_data_vl.jsonl에서 직접 생성")
    args = parser.parse_args()

    if args.model == "all":
        all_dirs = []
        for dirs in MODEL_DIRS.values():
            all_dirs.extend(dirs)
        output_path = args.output or "./data/ft_train_vl_all.jsonl"
        if args.from_jsonl:
            print("⚠️  --from-jsonl 은 단일 모델에서만 지원됩니다.")
        else:
            preprocess_vl(all_dirs, TRAINING_DATA_DIRS["bu"], output_path)
    else:
        output_path = args.output or f"./data/ft_train_vl_{args.model}_v2.jsonl"
        if args.from_jsonl:
            preprocess_from_jsonl(TRAINING_DATA_DIRS[args.model], output_path)
        else:
            preprocess_vl(
                MODEL_DIRS[args.model],
                TRAINING_DATA_DIRS[args.model],
                output_path
            )