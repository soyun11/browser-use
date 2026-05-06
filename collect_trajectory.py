# collect_trajectory.py
# 충남대 도서관 AI Agent trajectory 수집 스크립트
# 동료 collect.py 구조 기반, 도서관 특화 버전

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from browser_use import BrowserSession, BrowserProfile, ChatBrowserUse
from logged_agent import LoggedAgent
from dotenv import load_dotenv
from tasks.library_tasks import (
    ALL_TASKS,
    TASK_CATEGORIES,
    CATEGORY_DOMAINS,
    LOGIN_CATEGORIES,
    LIBRARY_BASE_CONTEXT,
)

load_dotenv()

# ──────────────────────────────────────────
# 수집 설정
# ──────────────────────────────────────────
MAX_STEPS = 30
SLEEP_BETWEEN_TASKS = 3


# ──────────────────────────────────────────
# sensitive_data 설정
# ──────────────────────────────────────────
def get_sensitive_data(category: str) -> dict | None:
    cnu_id = os.getenv("CNU_ID")
    cnu_pw = os.getenv("CNU_PASSWORD")
    if not cnu_id or not cnu_pw or category not in LOGIN_CATEGORIES:
        return None
    return {
        "library.cnu.ac.kr":  {"x_user_id": cnu_id, "x_user_pw": cnu_pw},
        "dcs-lcms.cnu.ac.kr": {"x_user_id": cnu_id, "x_user_pw": cnu_pw},
        "portal.cnu.ac.kr":   {"x_user_id": cnu_id, "x_user_pw": cnu_pw},
    }


# ──────────────────────────────────────────
# trajectory 파싱
# ──────────────────────────────────────────
def parse_trajectory(task: str, category: str, history, total_elapsed: float) -> dict:
    steps = []
    for step_idx, h in enumerate(history.history):
        # 스텝별 소요 시간 (metadata에서 추출)
        step_duration = None
        if h.metadata and hasattr(h.metadata, "duration_seconds"):
            step_duration = round(h.metadata.duration_seconds, 2)

        step_data = {
            "step": step_idx + 1,
            "duration_seconds": step_duration,
            "evaluation": None,
            "memory": None,
            "next_goal": None,
            "actions": [],
            "results": [],
        }
        if h.model_output:
            step_data["evaluation"] = h.model_output.evaluation_previous_goal
            step_data["memory"] = h.model_output.memory
            step_data["next_goal"] = h.model_output.next_goal
            for action in h.model_output.action:
                action_dict = action.model_dump(exclude_none=True)
                for key, val in action_dict.items():
                    if val is not None:
                        step_data["actions"].append({"type": key, "params": val})
                        break
        for result in h.result:
            step_data["results"].append({
                "is_done": result.is_done,
                "success": result.success,
                "error": result.error,
                "memory": result.long_term_memory,
            })
        steps.append(step_data)

    final_result = None
    success = False
    for h in reversed(history.history):
        if h.model_output:
            for action in h.model_output.action:
                action_dict = action.model_dump(exclude_none=True)
                if "done" in action_dict:
                    final_result = action_dict["done"].get("text")
                    success = action_dict["done"].get("success", False)
                    break
        if final_result:
            break

    # history에서 총 시간 계산 (metadata 합산)
    history_duration = round(history.total_duration_seconds(), 2) if hasattr(history, "total_duration_seconds") else None

    return {
        "task": task,
        "category": category,
        "timestamp": datetime.now().isoformat(),
        "success": success,
        "total_steps": len(steps),
        "total_duration_seconds": history_duration or round(total_elapsed, 2),
        "steps": steps,
        "final_result": final_result,
    }


# ──────────────────────────────────────────
# 태스크 실행 (태스크마다 독립 브라우저 세션)
# ──────────────────────────────────────────
async def run_task(task_info: dict, task_index: int, output_dir: Path, data_dir: Path) -> dict:
    raw_task = task_info["task"]
    category = task_info["category"]
    use_base_context = task_info.get("use_base_context", True)

    # BASE_CONTEXT 붙이기 (도서관 단독 태스크만)
    full_task = f"{LIBRARY_BASE_CONTEXT} {raw_task}" if use_base_context else raw_task

    print(f"\n{'='*60}")
    print(f"[{task_index+1}/{len(ALL_TASKS)}] [{category}] {raw_task[:50]}...")
    print('='*60)

    allowed_domains = CATEGORY_DOMAINS.get(category, ["library.cnu.ac.kr"])
    sensitive_data = get_sensitive_data(category)

    browser_session = BrowserSession(
        browser_profile=BrowserProfile(
            headless=True,
            allowed_domains=allowed_domains,
        )
    )

    # 수정 - 이렇게 교체
    agent = LoggedAgent(
        task=full_task,
        llm=ChatBrowserUse(),
        log_path=str(data_dir / "training_data.jsonl"),
        save_conversation_path=str(data_dir / "conversations" / f"task_{task_index:03d}.json"),
        generate_gif=str(data_dir / "gifs" / f"task_{task_index:03d}.gif"),
        browser=browser_session,
        sensitive_data=sensitive_data,
    )

    result = {
        "task_index": task_index,
        "task": raw_task,
        "category": category,
        "success": None,
        "error": None,
        "steps": None,
        "final_result": None,
    }

    try:
        start_time = asyncio.get_event_loop().time()
        history = await agent.run(max_steps=MAX_STEPS)
        elapsed = asyncio.get_event_loop().time() - start_time

        trajectory = parse_trajectory(raw_task, category, history, elapsed)

        # trajectory JSON 저장 (개별 파일)
        output_path = output_dir / f"task_{task_index:03d}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(trajectory, f, ensure_ascii=False, indent=2)

        # trajectory JSONL 저장 (전체 누적, 한 줄에 태스크 하나)
        jsonl_path = data_dir / "trajectories.jsonl"
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(trajectory, ensure_ascii=False) + "\n")

        result["success"] = trajectory["success"]
        result["steps"] = trajectory["total_steps"]
        result["total_duration_seconds"] = trajectory["total_duration_seconds"]
        result["final_result"] = trajectory["final_result"]

        status = "✅ 성공" if result["success"] else "❌ 실패"
        print(f"결과: {status} ({result['steps']} steps, {result['total_duration_seconds']}초)")
        print(f"결과: {status} ({result['steps']} steps)")

    except Exception as e:
        result["success"] = False
        result["error"] = str(e)
        print(f"에러: {e}")

        error_path = output_dir / f"task_{task_index:03d}_error.json"
        with open(error_path, "w", encoding="utf-8") as f:
            json.dump({
                "task": raw_task, "category": category,
                "timestamp": datetime.now().isoformat(),
                "success": False, "error": str(e),
            }, f, ensure_ascii=False, indent=2)

    finally:
        await browser_session.stop()

    await asyncio.sleep(SLEEP_BETWEEN_TASKS)
    return result


# ──────────────────────────────────────────
# 메인
# ──────────────────────────────────────────
async def main():
    # 디렉토리 생성
    output_dir = Path("trajectories")
    data_dir = Path("data")
    for d in [output_dir, data_dir, data_dir / "conversations",
              data_dir / "gifs", data_dir / "results"]:
        d.mkdir(exist_ok=True)

    # 실행 범위 설정
    # 방법1) 환경변수: START_TASK=0 END_TASK=10 python collect_trajectory.py
    # 방법2) 커맨드라인: python collect_trajectory.py 0 10
    start = int(os.getenv("START_TASK", sys.argv[1] if len(sys.argv) > 1 else "0"))
    end   = int(os.getenv("END_TASK",   sys.argv[2] if len(sys.argv) > 2 else str(len(ALL_TASKS))))

    print(f"총 {end - start}개 태스크 수집 시작 (index {start}~{end-1})")
    print(f"전체 태스크 수: {len(ALL_TASKS)}개")

    # 로그인 필요 태스크 경고
    login_tasks = [ALL_TASKS[i] for i in range(start, end)
                   if ALL_TASKS[i]["category"] in LOGIN_CATEGORIES]
    if login_tasks:
        cnu_id = os.getenv("CNU_ID")
        if not cnu_id:
            print("⚠️  경고: .env에 CNU_ID/CNU_PASSWORD가 없습니다. 로그인 시나리오가 실패할 수 있습니다.")
        else:
            print(f"🔑 로그인 계정 확인: {cnu_id} ({len(login_tasks)}개 로그인 태스크 포함)")

    all_results = []
    for i in range(start, end):
        result = await run_task(ALL_TASKS[i], i, output_dir, data_dir)
        all_results.append(result)

        # 10개마다 중간 저장
        if len(all_results) % 10 == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(data_dir / "results" / f"progress_{timestamp}.json", "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            print(f"💾 중간 저장 완료 ({len(all_results)}개)")

    # ── 최종 요약 ──
    print(f"\n{'='*60}")
    print("수집 완료 요약")
    print('='*60)

    success_count = sum(1 for r in all_results if r["success"])
    fail_count = len(all_results) - success_count
    print(f"성공: {success_count}개 / 실패: {fail_count}개 / 성공률: {success_count/len(all_results)*100:.1f}%")

    # 카테고리별 성공률
    cat_stats = {}
    for r in all_results:
        cat = r["category"]
        if cat not in cat_stats:
            cat_stats[cat] = {"success": 0, "fail": 0}
        if r["success"]:
            cat_stats[cat]["success"] += 1
        else:
            cat_stats[cat]["fail"] += 1

    print("\n카테고리별 성공률:")
    for cat, stats in cat_stats.items():
        total = stats["success"] + stats["fail"]
        rate = stats["success"] / total * 100 if total > 0 else 0
        print(f"  [{cat}] {stats['success']}/{total} ({rate:.0f}%)")

    # 실패 태스크 목록
    if fail_count > 0:
        print("\n실패 태스크:")
        for r in all_results:
            if not r["success"]:
                print(f"  [{r['task_index']}] [{r['category']}] {r['task'][:60]}")
                if r.get("error"):
                    print(f"       에러: {r['error'][:100]}")

    # 최종 저장
    with open(data_dir / "results" / "final_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n결과 저장: data/results/final_results.json")
    print(f"trajectory: trajectories/")
    print(f"학습 데이터: data/training_data.jsonl")


if __name__ == "__main__":
    asyncio.run(main())