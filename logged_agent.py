# logged_agent.py
import json
from browser_use import Agent
from browser_use.llm.messages import BaseMessage
from browser_use.agent.views import AgentOutput


class LoggedAgent(Agent):
    def __init__(self, *args, log_path="training_data.jsonl", **kwargs):
        super().__init__(*args, **kwargs)
        self.log_path = log_path

    def _log_training_data(self, input_messages: list[BaseMessage], parsed: AgentOutput):
        """매 LLM 호출마다 input/output 쌍을 JSONL로 저장"""
        try:
            data = {
                "input": [m.model_dump() for m in input_messages],
                "output": parsed.model_dump()
            }
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
        except Exception as e:
            self.logger.warning(f"Failed to log training data: {e}")

    async def get_model_output(self, input_messages):
        parsed = await super().get_model_output(input_messages)
        self._log_training_data(input_messages, parsed)
        return parsed