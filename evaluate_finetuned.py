# evaluate_finetuned.py
# 파인튜닝된 모델(vLLM 서버)로 도서관/증명서 시나리오 실행 및 성능 평가
# 사용법:
#   python evaluate_finetuned.py 0 100           # 도서관 0~99
#   python evaluate_finetuned.py --cert 0 100    # 증명서 0~99

import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

from browser_use import BrowserSession, BrowserProfile
from logged_agent import LoggedAgent
from dotenv import load_dotenv
from tasks.library_tasks import (
    ALL_TASKS as LIBRARY_TASKS,
    CATEGORY_DOMAINS as LIBRARY_CATEGORY_DOMAINS,
    LOGIN_CATEGORIES as LIBRARY_LOGIN_CATEGORIES,
    LIBRARY_BASE_CONTEXT,
)
from tasks.cert_tasks import (
    ALL_TASKS as CERT_TASKS,
    CATEGORY_DOMAINS as CERT_CATEGORY_DOMAINS,
    LOGIN_CATEGORIES as CERT_LOGIN_CATEGORIES,
    CERT_BASE_CONTEXT,
)

load_dotenv()

# ──────────────────────────────────────────
# 설정
# ──────────────────────────────────────────
MAX_STEPS = 30
SLEEP_BETWEEN_TASKS = 3

# vLLM 서버 설정
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://서버IP:8000/v1")
VLLM_MODEL = "./output/qwen2.5-32b-cnu-merged"


# ──────────────────────────────────────────
# 파인튜닝 모델 LLM 로드
# ──────────────────────────────────────────
def get_finetuned_llm():
    from browser_use.llm.openai.chat import ChatOpenAI
    return ChatOpenAI(
        model=VLLM_MODEL,
        base_url=VLLM_BASE_URL,
        api_key="dummy",
    )


# ──────────────────────────────────────────
# sensitive_data 설정
# ──────────────────────────────────────────
def get_sensitive_data(category: str, login_categories: set) -> dict | None:
    cnu_id = os.getenv("CNU_ID")
    cnu_pw = os.getenv("CNU_PASSWORD")
    if not cnu_id or not cnu_pw or category not in login_categories:
        return None
    return {
        "library.cnu.ac.kr":  {"x_user_id": cnu_id, "x_user_pw": cnu_pw},
        "dcs-lcms.cnu.ac.kr": {"x_user_id": cnu_id, "x_user_pw": cnu_pw},
        "portal.cnu.ac.kr":   {"x_user_id": cnu_id, "x_user_pw": cnu_pw},
        "cnu.icerti.com":     {"x_user_id": cnu_id, "x_user_pw": cnu_pw},
    }


# ──────────────────────────────────────────
# trajectory 파싱
# ──────────────────────────────────────────
def parse_trajectory(task: str, category: str, history, total_elapsed: float) -> dict:
    steps = []
    for step_idx, h in enumerate(history.history):
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
# 태스크 실행
# ──────────────────────────────────────────
async def run_task(
    task_info: dict,
    task_index: int,
    llm,
    output_dir: Path,
    data_dir: Path,
    all_tasks: list,
    base_context: str,
    category_domains: dict,
    login_categories: set,
    is_cert: bool,
) -> dict:
    raw_task = task_info["task"]
    category = task_info["category"]
    use_base_context = task_info.get("use_base_context", True)
    full_task = f"{base_context} {raw_task}" if use_base_context else raw_task

    print(f"\n{'='*60}")
    print(f"[{task_index+1}/{len(all_tasks)}] [{category}] {raw_task[:50]}...")
    print('='*60)

    allowed_domains = category_domains.get(category, ["library.cnu.ac.kr"])
    sensitive_data = get_sensitive_data(category, login_categories)
    task_prefix = "cert_task" if is_cert else "task"

    browser_session = BrowserSession(
        browser_profile=BrowserProfile(
            headless=True,
            allowed_domains=allowed_domains,
        )
    )

    agent = LoggedAgent(
        task=full_task,
        llm=llm,
        use_vision=False,
        log_path=str(data_dir / "training_data_finetuned.jsonl"),
        save_conversation_path=str(data_dir / "conversations_finetuned" / f"{task_prefix}_{task_index:03d}.json"),
        generate_gif=str(data_dir / "gifs_finetuned" / f"{task_prefix}_{task_index:03d}.gif"),
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
        "total_duration_seconds": None,
        "final_result": None,
    }

    try:
        start_time = asyncio.get_event_loop().time()
        history = await agent.run(max_steps=MAX_STEPS)
        elapsed = asyncio.get_event_loop().time() - start_time

        trajectory = parse_trajectory(raw_task, category, history, elapsed)

        # 개별 JSON 저장
        output_path = output_dir / f"{task_prefix}_{task_index:03d}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(trajectory, f, ensure_ascii=False, indent=2)

        # JSONL 누적 저장
        jsonl_path = data_dir / "trajectories_finetuned.jsonl"
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(trajectory, ensure_ascii=False) + "\n")

        result["success"] = trajectory["success"]
        result["steps"] = trajectory["total_steps"]
        result["total_duration_seconds"] = trajectory["total_duration_seconds"]
        result["final_result"] = trajectory["final_result"]

        status = "✅ 성공" if result["success"] else "❌ 실패"
        print(f"결과: {status} ({result['steps']} steps, {result['total_duration_seconds']}초)")

    except Exception as e:
        result["success"] = False
        result["error"] = str(e)
        print(f"에러: {e}")

        error_path = output_dir / f"{task_prefix}_{task_index:03d}_error.json"
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--cert", action="store_true", help="증명서 태스크 실행 (기본: 도서관)")
    parser.add_argument("start", type=int, nargs="?", default=0)
    parser.add_argument("end", type=int, nargs="?", default=None)
    args = parser.parse_args()

    # 태스크 및 설정 선택
    if args.cert:
        all_tasks = CERT_TASKS
        base_context = CERT_BASE_CONTEXT
        category_domains = CERT_CATEGORY_DOMAINS
        login_categories = CERT_LOGIN_CATEGORIES
        output_dir = Path("cert_trajectories_finetuned")
    else:
        all_tasks = LIBRARY_TASKS
        base_context = LIBRARY_BASE_CONTEXT
        category_domains = LIBRARY_CATEGORY_DOMAINS
        login_categories = LIBRARY_LOGIN_CATEGORIES
        output_dir = Path("trajectories_finetuned")

    start = args.start
    end = args.end or len(all_tasks)
    data_dir = Path("data")

    # 디렉토리 생성
    for d in [output_dir, data_dir,
              data_dir / "conversations_finetuned",
              data_dir / "gifs_finetuned",
              data_dir / "results"]:
        d.mkdir(parents=True, exist_ok=True)

    # LLM 로드
    print(f"\n🤖 모델: 파인튜닝 모델 (vLLM @ {VLLM_BASE_URL})")
    llm = get_finetuned_llm()

    print(f"총 {end - start}개 태스크 수집 시작 (index {start}~{end-1})")
    print(f"전체 태스크 수: {len(all_tasks)}개")
    print(f"저장 경로: {output_dir}/")

    # 로그인 필요 태스크 경고
    login_tasks = [all_tasks[i] for i in range(start, end)
                   if all_tasks[i]["category"] in login_categories]
    if login_tasks:
        cnu_id = os.getenv("CNU_ID")
        if not cnu_id:
            print("⚠️  경고: .env에 CNU_ID/CNU_PASSWORD가 없습니다.")
        else:
            print(f"🔑 로그인 계정 확인: {cnu_id} ({len(login_tasks)}개 로그인 태스크 포함)")

    all_results = []
    for i in range(start, end):
        result = await run_task(
            all_tasks[i], i, llm, output_dir, data_dir, all_tasks,
            base_context, category_domains, login_categories, args.cert,
        )
        all_results.append(result)

        # 10개마다 중간 저장
        if len(all_results) % 10 == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cert_prefix = "cert_" if args.cert else ""
            with open(data_dir / "results" / f"{cert_prefix}progress_finetuned_{timestamp}.json", "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            print(f"💾 중간 저장 완료 ({len(all_results)}개)")

    # ── 최종 요약 ──
    print(f"\n{'='*60}")
    print("수집 완료 요약 [파인튜닝 모델]")
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
    cert_prefix = "cert_" if args.cert else ""
    with open(data_dir / "results" / f"{cert_prefix}final_finetuned.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n결과 저장: data/results/{cert_prefix}final_finetuned.json")
    print(f"trajectory: {output_dir}/")


if __name__ == "__main__":
    asyncio.run(main())