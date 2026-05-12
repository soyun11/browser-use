#!/usr/bin/env python3
"""
training_data_gemini.jsonl (raw) → 텍스트 전용 파인튜닝 포맷 변환
- browser_state (DOM 텍스트) 포함
- 이미지 제거
- input/output → messages 형식으로 변환
"""

import json
import argparse
from pathlib import Path

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


def extract_text_from_content(content) -> str:
    """content에서 텍스트만 추출 (이미지 제거)"""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    texts.append(part["text"])
                # image_url 타입은 무시
        return "\n".join(texts)
    return ""


def convert_output_to_action(output: dict) -> str:
    """output dict → 파인튜닝 포맷 JSON 문자열"""
    action = output.get("action", [])
    # action 형식 변환: [{"navigate": {...}}] → [{"type": "navigate", "params": {...}}]
    converted_actions = []
    for a in action:
        if isinstance(a, dict):
            for action_type, params in a.items():
                converted_actions.append({
                    "type": action_type,
                    "params": params if params else {}
                })
    
    result = {
        "memory": output.get("memory", ""),
        "next_goal": output.get("next_goal", ""),
        "actions": converted_actions
    }
    return json.dumps(result, ensure_ascii=False)


def process_sample(data: dict) -> dict | None:
    """raw 샘플 → messages 형식으로 변환"""
    input_msgs = data.get("input", [])
    output = data.get("output", {})
    
    if not input_msgs or not output:
        return None
    
    messages = []
    
    # system 메시지
    messages.append({
        "role": "system",
        "content": SYSTEM_PROMPT
    })
    
    # input 메시지들 처리
    for msg in input_msgs:
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        if role == "system":
            # system은 이미 위에서 처리
            continue
        elif role == "user":
            text = extract_text_from_content(content)
            if text.strip():
                messages.append({
                    "role": "user",
                    "content": text
                })
        elif role == "assistant":
            text = extract_text_from_content(content)
            if text.strip():
                messages.append({
                    "role": "assistant",
                    "content": text
                })
    
    # 마지막 assistant 메시지 (output)
    action_str = convert_output_to_action(output)
    messages.append({
        "role": "assistant",
        "content": action_str
    })
    
    # 최소 3개 메시지 (system + user + assistant) 필요
    if len(messages) < 3:
        return None
    
    return {"messages": messages}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/training_data_gemini.jsonl")
    parser.add_argument("--output", default="data/ft_train_text_gemini.jsonl")
    parser.add_argument("--min-messages", type=int, default=3)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    used = 0
    skipped = 0

    with open(input_path, encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                total += 1
                result = process_sample(data)
                if result and len(result["messages"]) >= args.min_messages:
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                    used += 1
                else:
                    skipped += 1
            except Exception as e:
                skipped += 1
                continue

    print(f"총 {total}개 처리")
    print(f"사용: {used}개 / 스킵: {skipped}개")
    print(f"저장: {output_path}")

    # 샘플 확인
    print("\n=== 샘플 확인 ===")
    with open(output_path, encoding="utf-8") as f:
        sample = json.loads(f.readline())
    for msg in sample["messages"]:
        content = msg["content"]
        print(f"[{msg['role']}]: {content[:200]}")
        print("---")


if __name__ == "__main__":
    main()