import warnings
# Robustly suppress Pydantic serializer warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", message=".*PydanticSerializationUnexpectedValue.*")

import os
import subprocess
import sys
import asyncio
from dotenv import load_dotenv
from google.adk.models.lite_llm import LiteLlm
from google.adk.agents import SequentialAgent, LlmAgent
from google.adk.sessions import InMemorySessionService
from app.agents.utils.run_agent_query import run_agent_query, format_tool_events
from app.tools.mock_toolkit import printA, printB, printC

session_service = InMemorySessionService()
user_id = "default_user"

VLLM_API_BASE = os.environ.get("VLLM_API_BASE", "http://localhost:8000/v1")
VLLM_MODEL = "openai/gpt-oss-20b"

base_model = LiteLlm(
    model=f"openai/{VLLM_MODEL}",
    api_base=VLLM_API_BASE,
    api_key="sk-1234", # Mock key
    timeout=120,)

test_agent = LlmAgent(
    name = "test_agent",
    model = base_model,
    tools = [printA, printB, printC],
    instruction="""一定要使用工具，我要測試你使用工具的能力。
    Available Tools:
        - printA(): Returns a verification string for A
        - printB(): Returns a verification string for B
        - printC(): Returns a verification string for C

    Task:
    1. Call printA()
    2. Call printB()
    3. Call printC()
    4. Compile the EXACT return values from all three tools into the final report.
    
    CRITICAL: Do not invent outputs. You must wait for the tool execution and use the actual returned string.
    注意：請準確輸出工具名稱，不要自行添加任何後綴或標記（如 <|channel|> 等）。"""
    )

async def run_test_generation(query: str):
    agent = test_agent
    session = await session_service.create_session(app_name=agent.name, user_id=user_id)
    
    print("*"*50)
    print(f"🎤 Query: {query}")
    print("*"*50)
    print("🚀 Agent started...")
    
    result = await run_agent_query(agent, query, session, user_id, session_service)
    return result

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = sys.argv[1]
    else:
        query = "依序呼叫工具A、B、C，並把結果寫在最後。"
    for i in range(5):
        print(f"--- Attempt {i+1}/10 ---")
        result = asyncio.run(run_test_generation(query))
        
        if "An error occurred" in result.output:
            print(f"⚠️ Attempt {i+1} failed. Retrying with error feedback...")
            query += f"\n\nPrevious attempt failed with error:\n{result.output}\nPlease fix your tool usage."
        else:
            print(f"✅ Attempt {i+1} successful!")
            break
    
    print("\n✅ Final Markdown Output:\n")
    print(result.output)
    print("-"*50 + "\n")
    # print(result.think)
    print("-"*50 + "\n")
    events_log, tool_status = format_tool_events(result.events)
    # print(events_log)
    print("Tool Status Dictionary:\n", tool_status)
    print(" !!!!!!!!! complete !!!!!!!!!!")
