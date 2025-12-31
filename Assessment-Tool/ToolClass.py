import json
import random
from typing import Dict, List, Any
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.callbacks import BaseCallbackHandler
from tool_function import Search_Materials_Tool, Update_Config
from prompt_template import custom_prompt
from bayes_config import Config
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor

class MemoryManager:
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}

    def update_session(self, session_id: str, messages: List[Any]):
        if session_id not in self.sessions:
            self.sessions[session_id] = {"messages": [], "tool_calls": []}
        self.sessions[session_id]["messages"].extend(messages)
        for msg in messages:
            if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls"):
                self.sessions[session_id]["tool_calls"].extend(msg.tool_calls)

    def get_state(self, session_id: str) -> Dict:
        return self.sessions.get(session_id, {})

class ToolCallValidator(BaseCallbackHandler):
    def __init__(self):
        self.tool_triggered = None

    def on_tool_start(self, serialized, input_str, **kwargs):
        self.tool_triggered = serialized["name"]
