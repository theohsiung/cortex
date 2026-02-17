"""Base agent module wrapping Google ADK agents."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from app.task.task_manager import TaskManager

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from google.adk.sessions import InMemorySessionService


@dataclass
class ExecutionContext:
    """Context for a single step execution, isolating state for parallel execution."""

    step_index: int
    pending_calls: dict[str, dict] = field(default_factory=dict)


@dataclass
class AgentResult:
    """Result from agent execution."""

    events: list[Any]
    output: str
    is_complete: bool = True


class BaseAgent:
    """
    Base class for all agents, wraps any Google ADK agent.

    Supports: LlmAgent, LoopAgent, SequentialAgent, ParallelAgent, etc.

    Usage:
        # With LlmAgent
        from google.adk.agents import LlmAgent
        llm_agent = LlmAgent(name="my_agent", model=model, tools=[...])
        base = BaseAgent(agent=llm_agent, tool_functions={...})

        # With LoopAgent
        from google.adk.agents import LoopAgent
        loop_agent = LoopAgent(name="loop", sub_agents=[agent1, agent2])
        base = BaseAgent(agent=loop_agent, tool_functions={...})
    """

    def __init__(
        self,
        agent: Any,
        tool_functions: dict | None = None,
        plan_id: str | None = None,
    ) -> None:
        """Initialize BaseAgent with a pre-built ADK agent.

        Args:
            agent: Any ADK agent (LlmAgent, LoopAgent, SequentialAgent, etc.).
            tool_functions: Dict mapping tool names to callable functions.
            plan_id: Optional plan ID for TaskManager integration.
        """
        self.agent = agent
        self.tool_functions = tool_functions or {}

        # Plan integration
        self.plan_id = plan_id
        self.plan = TaskManager.get_plan(plan_id) if plan_id else None

        # Event tracking
        self._tool_events: list[dict] = []

        # Session service (lazy init)
        self._session_service: InMemorySessionService | None = None

    def _get_session_service(self) -> InMemorySessionService:
        """Lazy initialization of session service."""
        if self._session_service is None:
            from google.adk.sessions import InMemorySessionService

            self._session_service = InMemorySessionService()
        return self._session_service

    async def execute(
        self,
        query: str,
        max_iteration: int = 10,
        exec_context: ExecutionContext | None = None,
    ) -> AgentResult:
        """Execute query with automatic retry loop."""
        session_service = self._get_session_service()
        session = await session_service.create_session(app_name=self.agent.name, user_id="default")

        last_error = None
        for i in range(max_iteration):
            try:
                result = await self._run_once(query, session, exec_context)
            except json.JSONDecodeError as e:
                last_error = e
                logger.warning(
                    "LLM returned malformed JSON, retrying (%s/%s)",
                    i + 1,
                    max_iteration,
                )
                continue
            if result.is_complete:
                return result

        if last_error is not None:
            raise last_error
        return self._handle_max_iteration()

    async def _run_once(
        self,
        query: str,
        session: Any,
        exec_context: ExecutionContext | None = None,
    ) -> AgentResult:
        """Single execution run."""
        from google.adk.runners import Runner
        from google.genai.types import Content, Part

        runner = Runner(
            agent=self.agent, session_service=self._get_session_service(), app_name=self.agent.name
        )

        events = []
        final_output = ""

        async for event in runner.run_async(
            user_id="default",
            session_id=session.id,
            new_message=Content(parts=[Part(text=query)], role="user"),
        ):
            events.append(event)
            self._process_event(event, exec_context)

            if event.is_final_response() and event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        final_output += part.text

        # Check if complete (no pending tool calls)
        is_complete = True
        for event in events:
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, "function_call") and part.function_call:
                        name = part.function_call.name
                        if name is not None:
                            is_complete = self._has_tool_response(events, name)

        return AgentResult(events=events, output=final_output, is_complete=is_complete)

    def _process_event(self, event: Any, exec_context: ExecutionContext | None = None) -> None:
        """Process event and track tool calls."""
        if not event.content or not event.content.parts:
            return

        for part in event.content.parts:
            if hasattr(part, "function_call") and part.function_call:
                call = part.function_call
                call_time = datetime.now().isoformat()
                pending_calls = exec_context.pending_calls if exec_context else {}
                call_id = f"{call.name}_{len(pending_calls)}"
                call_args = dict(call.args) if call.args else {}
                pending_calls[call_id] = {
                    "tool": call.name,
                    "args": call_args,
                    "timestamp": call_time,
                }
                self._track_tool_event({"type": "call", "name": call.name, "args": call.args})
                # Record pending call to plan for verification
                self._record_pending_call(call.name, call_args, call_time, exec_context)
            elif hasattr(part, "function_response") and part.function_response:
                resp = part.function_response
                self._track_tool_event({"type": "response", "name": resp.name})
                # Record to plan if step is active
                self._record_tool_to_plan(resp.name, resp.response, exec_context)

    def _has_tool_response(self, events: list, tool_name: str) -> bool:
        """Check if tool call has a response."""
        for event in events:
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, "function_response") and part.function_response:
                        if part.function_response.name == tool_name:
                            return True
        return False

    def _handle_max_iteration(self) -> AgentResult:
        """Handle max iteration reached."""
        return AgentResult(events=[], output="Max iterations reached", is_complete=True)

    def _track_tool_event(self, event: dict) -> None:
        """Track tool call/response event."""
        self._tool_events.append(event)

    def _record_pending_call(
        self,
        tool_name: str,
        args: dict,
        call_time: str,
        exec_context: ExecutionContext | None = None,
    ) -> None:
        """Record pending tool call to plan for verification tracking."""
        if self.plan is None or exec_context is None:
            return

        self.plan.add_tool_call_pending(
            step_index=exec_context.step_index, tool=tool_name, args=args, call_time=call_time
        )

    def _record_tool_to_plan(
        self,
        tool_name: str,
        result: Any,
        exec_context: ExecutionContext | None = None,
    ) -> None:
        """Update pending tool call to success in plan's step_tool_history."""
        if self.plan is None or exec_context is None:
            return

        step_index = exec_context.step_index
        pending_calls = exec_context.pending_calls
        response_time = datetime.now().isoformat()

        # Find matching pending call
        matching_call = None
        matching_id = None
        for call_id, call_info in pending_calls.items():
            if call_info["tool"] == tool_name:
                matching_call = call_info
                matching_id = call_id
                break

        if matching_call is None or matching_id is None:
            return

        # Update pending call to success in plan
        self.plan.update_tool_result(
            step_index=step_index, tool=tool_name, result=result, response_time=response_time
        )

        # Extract and record files
        files = self._extract_files(tool_name, matching_call["args"], result)
        for file_path in files:
            self.plan.add_file(step_index, file_path)

        # Remove from pending
        del pending_calls[matching_id]

    def _extract_files(self, tool_name: str, args: dict, result: Any) -> list[str]:
        """Extract file paths from tool call."""
        files = []

        # Explicit file operations
        if tool_name in ("write_file", "create_directory") and "path" in args:
            files.append(args["path"])

        # Shell redirect patterns: > file.txt or >> file.txt
        if tool_name == "run_command" and "command" in args:
            cmd = args["command"]
            # Match > or >> followed by filename (simple pattern)
            redirect_matches = re.findall(r">{1,2}\s*([^\s;&|]+)", cmd)
            files.extend(redirect_matches)

        # Python open() patterns
        if tool_name == "run_python" and "code" in args:
            code = args["code"]
            # Match open('file', 'w') or open("file", "w") patterns
            open_matches = re.findall(r"open\s*\(\s*['\"]([^'\"]+)['\"].*['\"][wax]", code)
            files.extend(open_matches)

        return files

    def get_tool_summary(self) -> dict:
        """Get tool usage statistics."""
        calls = [e for e in self._tool_events if e["type"] == "call"]
        responses = [e for e in self._tool_events if e["type"] == "response"]

        return {
            "total_calls": len(calls),
            "total_responses": len(responses),
            "tools_used": list(set(e["name"] for e in calls)),
        }

    @staticmethod
    def should_include_aliases(model: Any) -> bool:
        """Check if model supports aliased tool names with special characters.

        Gemini API doesn't support special chars like <|channel|> in function names.
        gpt-oss models may hallucinate these suffixes and need aliases.

        Args:
            model: The LLM model instance.

        Returns:
            True if model needs aliased tool names, False otherwise.
        """
        if model is None:
            return False
        model_str = str(model).lower()
        # Gemini doesn't support special chars in function names
        if "gemini" in model_str:
            return False
        # gpt-oss models may need aliases for hallucinated tool names
        if "gpt-oss" in model_str or "openai" in model_str:
            return True
        return False
