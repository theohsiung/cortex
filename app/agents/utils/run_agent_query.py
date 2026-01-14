import warnings
# Robustly suppress Pydantic serializer warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", message=".*PydanticSerializationUnexpectedValue.*")

import re
import asyncio
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple
try:
    from IPython.display import display, Markdown
except ImportError:
    display = None
    Markdown = None
import google.genai
from google.adk.agents import Agent, SequentialAgent, LoopAgent, ParallelAgent
from google.adk.tools import google_search, ToolContext
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, Session
from google.genai.types import Content, Part
from getpass import getpass
import os
from dotenv import load_dotenv
load_dotenv()
import logging

@dataclass
class AgentResult:
    events: list[Any]
    think: str
    output: str

async def run_agent_query(agent: Agent, query: str, session: Session, user_id: str, session_service, is_router: bool = False):
    """Initializes a runner and executes a query for a given agent and session."""
    print(f"\n🚀 Running query for agent: '{agent.name}' in session: '{session.id}'...")
    # Compute the module-derived app name for diagnostics, but use `agent.name`
    # as the runner app_name so it matches sessions created with that name.
    try:
        module_app_name = agent.__class__.__module__
        module_app_name_short = module_app_name.split('.')[-1]
    except Exception:
        module_app_name_short = None
    logging.debug(f"agent.name='{agent.name}', module_app_name='{module_app_name_short}'")

    if module_app_name_short and module_app_name_short != agent.name:
        logging.debug(
            "App name mismatch diagnostic: agent.name='%s', module suggests '%s'."
            % (agent.name, module_app_name_short)
        )

    # Use the explicit agent.name for the runner so it can locate sessions
    runner = Runner(
        agent=agent,
        session_service=session_service,
        app_name=agent.name
    )

    events = []
    final_think = ""
    final_response = ""
    
    try:
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session.id,
            new_message=Content(parts=[Part(text=query)], role="user")
        ):
            events.append(event)
            if not is_router:
                pass
            
            # Extract thoughts from ALL events
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, 'thought') and part.thought:
                        final_think += part.text if part.text else ""

            if event.is_final_response():
                # 遍歷所有 parts，找到非 thought 的實際回答
                for part in event.content.parts:
                    # thought=True 的是思考過程，已經在上面處理過了，這裡只要確保不加入 final_response
                    if hasattr(part, 'thought') and part.thought:
                        continue
                    # 檢查是否有 text 屬性
                    if hasattr(part, 'text') and part.text:
                        final_response += part.text # support multiple text parts if any

    except Exception as e:
        final_response = f"An error occurred: {e}"

    if not is_router:
        def _running_in_jupyter() -> bool:
            try:
                from IPython import get_ipython
                ip = get_ipython()
                if ip is None:
                    return False
                return ip.__class__.__name__ == 'ZMQInteractiveShell'
            except Exception:
                return False

        if _running_in_jupyter():
            display(Markdown(final_response))
        else:
            pass

    return AgentResult(events=events, think=final_think, output=final_response)

def format_tool_events(events) -> Tuple[str, dict]:
    output_lines = []
    output_lines.append("\n" + "="*50)
    output_lines.append("📜 Event Log & Tool Usage Statistics")
    output_lines.append("="*50 + "\n")
    
    tool_calls = []
    tool_responses = []
    
    for idx, event in enumerate(events):
        output_lines.append(f"--- Event {idx} ---")
        if not event.content or not event.content.parts:
            output_lines.append("(No content)")
            continue
            
        for part in event.content.parts:
            if hasattr(part, 'function_call') and part.function_call:
                output_lines.append(f"🔧 Tool Call: {part.function_call.name}")
                output_lines.append(f"   Args: {part.function_call.args}")
                tool_calls.append({"name": part.function_call.name, "event_idx": idx})
            elif hasattr(part, 'function_response') and part.function_response:
                output_lines.append(f"✅ Tool Response: {part.function_response.name}") # Assuming name exists or implied
                output_lines.append(f"   Response: {part.function_response.response}")
                tool_responses.append({"name": part.function_response.name, "event_idx": idx})
            elif hasattr(part, 'text') and part.text:
                pass
                # output_lines.append(f"📝 Text: {part.text[:50]}...") # Optional: print preview of text

    output_lines.append("\n" + "="*50)
    output_lines.append("📊 Summary Statistics")
    output_lines.append("="*50)
    output_lines.append(f"Total Events: {len(events)}")
    output_lines.append(f"Total Tool Calls: {len(tool_calls)}")
    output_lines.append(f"Total Tool Responses: {len(tool_responses)}")
    output_lines.append("-" * 30)
    output_lines.append(f"{'Tool Name':<20} | {'Status':<10}")
    output_lines.append("-" * 30)
    
    # Simple matching strategy: assumption of sequential processing
    # or just separate counting if matching is hard without IDs.
    # In many frameworks, calls and responses are strictly ordered.
    
    # We will try to map calls to responses.
    # This logic assumes FIFO: first call to 'A' is matched by first response from 'A'.
    
    matched_responses = list(tool_responses)
    tool_status = {}
    
    for call in tool_calls:
        status_text = "❌ Pending/Failed"
        is_success = False
        
        # Find first matching response that hasn't been used (if name matches)
        # Note: function_response object might have 'name'. If not, we might need heuristic.
        # Assuming event.function_response has a 'name' field based on typical structures.
        
        found_idx = -1
        for i, resp in enumerate(matched_responses):
            if resp["name"] == call["name"]:
                status_text = "✅ Success"
                is_success = True
                found_idx = i
                break
        
        if found_idx != -1:
            matched_responses.pop(found_idx)
            
        output_lines.append(f"{call['name']:<20} | {status_text:<10}")
        tool_status[call['name']] = is_success
        
    output_lines.append("="*50 + "\n")
    return "\n".join(output_lines), tool_status