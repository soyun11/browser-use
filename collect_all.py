# collect_all.py
# collect_trajectory.py 기반, 모델을 파라미터로 선택 가능
# 사용법:
#   python collect_all.py --model bu 0 100       # ChatBrowserUse 도서관
#   python collect_all.py --model gpt 0 100      # GPT-4o-mini 도서관
#   python collect_all.py --model gemini 0 100   # Gemini 도서관
#   python collect_all.py --model gpt --cert 0 100   # GPT-4o-mini 증명서
#   python collect_all.py --model gemini --cert 0 100 # Gemini 증명서

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
# 수집 설정
# ──────────────────────────────────────────
MAX_STEPS = 30
SLEEP_BETWEEN_TASKS = 30


# ──────────────────────────────────────────
# LLM 선택
# ──────────────────────────────────────────
def get_llm(model_name: str):
    if model_name == "bu":
        from browser_use import ChatBrowserUse
        return ChatBrowserUse()

    elif model_name == "gpt":
        from browser_use.llm.openai.chat import ChatOpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY가 .env에 없습니다.")
        os.environ["OPENAI_API_KEY"] = api_key
        return ChatOpenAI(model="gpt-4o-mini")

    elif model_name == "gemini":
        from browser_use.llm.google.chat import ChatGoogle
        return ChatGoogle(model="gemini-2.5-flash")

    else:
        raise ValueError(f"지원하지 않는 모델: {model_name}")


# ──────────────────────────────────────────
# 모델별 저장 경로 설정
# ──────────────────────────────────────────
def get_dirs(model_name: str, is_cert: bool) -> dict:
    cert_prefix = "cert_" if is_cert else ""
    model_suffix = "" if model_name == "bu" else f"_{model_name}"
    task_prefix = "cert_task" if is_cert else "task"

    return {
        "output_dir": Path(f"{cert_prefix}trajectories{model_suffix}"),
        "data_dir": Path("data"),
        "training_data": Path(f"data/training_data{model_suffix}.jsonl"),
        "conversations": Path(f"data/conversations{model_suffix}"),
        "gifs": Path(f"data/gifs{model_suffix}"),
        "results": Path("data/results"),
        "task_prefix": task_prefix,
    }


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
# trajectory 파싱 (collect_trajectory.py와 동일)
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
    dirs: dict,
    all_tasks: list,
    base_context: str,
    category_domains: dict,
    login_categories: set,
    model_name: str,
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

    browser_session = BrowserSession(
        browser_profile=BrowserProfile(
            headless=True,
            allowed_domains=allowed_domains,
        )
    )

    task_prefix = dirs["task_prefix"]
    agent = LoggedAgent(
        task=full_task,
        llm=llm,
        log_path=str(dirs["training_data"]),
        save_conversation_path=str(dirs["conversations"] / f"{task_prefix}_{task_index:03d}.json"),
        generate_gif=str(dirs["gifs"] / f"{task_prefix}_{task_index:03d}.gif"),
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
        output_path = dirs["output_dir"] / f"{task_prefix}_{task_index:03d}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(trajectory, f, ensure_ascii=False, indent=2)

        # JSONL 누적 저장 (모델별 분리)
        jsonl_path = dirs["data_dir"] / f"trajectories_{model_name}.jsonl"
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

        error_path = dirs["output_dir"] / f"{task_prefix}_{task_index:03d}_error.json"
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
    parser.add_argument("--model", choices=["bu", "gpt", "gemini"], required=True,
                        help="bu=ChatBrowserUse / gpt=GPT-4o-mini / gemini=Gemini-2.5-flash")
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
    else:
        all_tasks = LIBRARY_TASKS
        base_context = LIBRARY_BASE_CONTEXT
        category_domains = LIBRARY_CATEGORY_DOMAINS
        login_categories = LIBRARY_LOGIN_CATEGORIES

    start = args.start
    end = args.end or len(all_tasks)
    dirs = get_dirs(args.model, args.cert)

    # 디렉토리 생성
    for d in [dirs["output_dir"], dirs["data_dir"],
              dirs["conversations"], dirs["gifs"], dirs["results"]]:
        d.mkdir(parents=True, exist_ok=True)

    # LLM 로드
    print(f"\n🤖 모델: {args.model.upper()}")
    llm = get_llm(args.model)

    print(f"총 {end - start}개 태스크 수집 시작 (index {start}~{end-1})")
    print(f"전체 태스크 수: {len(all_tasks)}개")
    print(f"저장 경로: {dirs['output_dir']}/")

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
            all_tasks[i], i, llm, dirs, all_tasks,
            base_context, category_domains, login_categories,
            args.model,
        )
        all_results.append(result)

        # 10개마다 중간 저장
        if len(all_results) % 10 == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cert_prefix = "cert_" if args.cert else ""
            with open(dirs["results"] / f"{cert_prefix}progress_{args.model}_{timestamp}.json", "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            print(f"💾 중간 저장 완료 ({len(all_results)}개)")

    # ── 최종 요약 ──
    print(f"\n{'='*60}")
    print(f"수집 완료 요약 [{args.model.upper()}]")
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
    with open(dirs["results"] / f"{cert_prefix}final_{args.model}.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n결과 저장: data/results/{cert_prefix}final_{args.model}.json")
    print(f"trajectory: {dirs['output_dir']}/")
    print(f"학습 데이터: {dirs['training_data']}")


if __name__ == "__main__":
    asyncio.run(main())