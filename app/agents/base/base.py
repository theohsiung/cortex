import os
import subprocess
import sys
import asyncio
from dotenv import load_dotenv
from google.adk.models.lite_llm import LiteLlm
from google.adk.agents import SequentialAgent
from google.adk.sessions import InMemorySessionService


load_dotenv()

VLLM_API_BASE = os.environ.get("VLLM_API_BASE", "http://localhost:8000/v1")
VLLM_MODEL = "openai/gpt-oss-20b"

base_model = LiteLlm(
    model=f"openai/{VLLM_MODEL}",
    api_base=VLLM_API_BASE,
    api_key="sk-1234", # Mock key
    timeout=120,)

class BaseAgent:
    def __init__(self, agent_instance: AgentInstance, llm: ChatLLM, functions: {}, plan_id: str = None):
        self.agent_instance = agent_instance
        self.llm = llm
        self.tools = []
        self.mcp_tools = []
        self.mcp_tools = get_mcp_tools(self.agent_instance.template.skills)
        for skill in self.agent_instance.template.skills:
            self.tools.extend(convert_skill_to_tool(skill.model_dump(), 'en'))
        self.tools.extend(convert_mcp_tools(self.mcp_tools))
        self.functions = functions
        self.history = []
        self.plan_id = plan_id
        self._tool_event_sequence = 0  # 工具事件序列
        self._file_saver_call_count = {}  # 記錄每個步驟的file_saver調用次數
        # Only set plan to None if it hasn't been set by subclass
        if not hasattr(self, 'plan'):
            self.plan = None  # Will be set by subclasses that have access to Plan
    
    def execute(self, message: List[Dict[str, Any]], step_index=None, max_iter:int = 10):
        pass
