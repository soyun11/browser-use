# logged_agent_vl.py
# VL 파인튜닝용 - 스크린샷(이미지) + 텍스트 함께 저장
# is_meaningful 자동 태깅 포함

import json
import time
from browser_use import Agent
from browser_use.llm.messages import BaseMessage
from browser_use.agent.views import AgentOutput


class LoggedAgentVL(Agent):
    def __init__(self, *args, log_path="training_data_vl.jsonl", task_index=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_path = log_path
        self.task_index = task_index
        self._pending_logs = []
        self._step_counter = 0
        self._step_start_time = None
        self._prev_actions = []  # 반복 액션 감지용

    def _is_meaningful(self, parsed: AgentOutput) -> bool:
        """의미 없는 스텝 자동 판별"""
        actions = parsed.action if parsed.action else []
        if not actions:
            return False

        action_types = []
        for action in actions:
            action_dict = action.model_dump(exclude_none=True)
            keys = list(action_dict.keys())
            if keys:
                action_types.append(keys[0])

        # done 액션 → 항상 의미 있음
        if "done" in action_types:
            return True

        # wait만 하는 스텝 → 의미 없음
        if action_types == ["wait"]:
            return False

        # 이전 스텝과 완전히 같은 액션 반복 → 의미 없음
        current = json.dumps(action_types, ensure_ascii=False)
        if self._prev_actions and current == self._prev_actions[-1]:
            return False

        return True

    def _log_training_data(self, input_messages: list[BaseMessage], parsed: AgentOutput):
        """매 LLM 호출마다 input/output 쌍을 버퍼에 저장 (이미지 포함)"""
        try:
            self._step_counter += 1

            # 스텝 소요 시간 계산
            step_time = None
            if self._step_start_time:
                step_time = round(time.time() - self._step_start_time, 2)
            self._step_start_time = time.time()

            # is_meaningful 자동 판별
            meaningful = self._is_meaningful(parsed)

            # 이전 액션 업데이트
            action_types = []
            for action in (parsed.action or []):
                action_dict = action.model_dump(exclude_none=True)
                keys = list(action_dict.keys())
                if keys:
                    action_types.append(keys[0])
            self._prev_actions.append(json.dumps(action_types, ensure_ascii=False))
            if len(self._prev_actions) > 3:
                self._prev_actions.pop(0)

            # 메시지 직렬화 (이미지 포함)
            serialized_messages = []
            for m in input_messages:
                msg_dict = m.model_dump()
                content = msg_dict.get("content", "")

                if isinstance(content, list):
                    processed_parts = []
                    for part in content:
                        if isinstance(part, dict):
                            if part.get("type") == "text":
                                processed_parts.append({
                                    "type": "text",
                                    "text": part.get("text", "")
                                })
                            elif part.get("type") == "image_url":
                                processed_parts.append({
                                    "type": "image_url",
                                    "image_url": part.get("image_url", {})
                                })
                        elif isinstance(part, str):
                            processed_parts.append({
                                "type": "text",
                                "text": part
                            })
                    msg_dict["content"] = processed_parts

                serialized_messages.append(msg_dict)

            data = {
                "task_index": self.task_index,
                "step_number": self._step_counter,
                "step_time_seconds": step_time,
                "is_meaningful": meaningful,
                "input": serialized_messages,
                "output": parsed.model_dump(),
            }
            self._pending_logs.append(data)

        except Exception as e:
            self.logger.warning(f"Failed to buffer VL training data: {e}")

    def flush_logs(self, success: bool):
        """태스크 완료 후 success 정보와 함께 저장"""
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                for data in self._pending_logs:
                    data["success"] = success
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")
            self.logger.info(f"✅ {len(self._pending_logs)}개 스텝 저장 완료 (meaningful: {sum(1 for d in self._pending_logs if d['is_meaningful'])}개)")
        except Exception as e:
            self.logger.warning(f"Failed to flush VL logs: {e}")
        finally:
            self._pending_logs = []
            self._step_counter = 0
            self._step_start_time = None
            self._prev_actions = []

    async def get_model_output(self, input_messages):
        parsed = await super().get_model_output(input_messages)
        self._log_training_data(input_messages, parsed)
        return parsed