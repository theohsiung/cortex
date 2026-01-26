import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, TYPE_CHECKING

from app.task.task_manager import TaskManager

if TYPE_CHECKING:
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai.types import Content, Part


@dataclass
class ExecutionContext:
    """Context for a single step execution, isolating state for parallel execution"""
    step_index: int
    pending_calls: dict[str, dict] = field(default_factory=dict)


@dataclass
class AgentResult:
    """Result from agent execution"""
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
        tool_functions: dict = None,
        plan_id: str = None
    ):
        """
        Initialize BaseAgent with a pre-built ADK agent.

        Args:
            agent: Any ADK agent (LlmAgent, LoopAgent, SequentialAgent, etc.)
            tool_functions: Dict mapping tool names to callable functions
            plan_id: Optional plan ID for TaskManager integration
        """
        self.agent = agent
        self.tool_functions = tool_functions or {}

        # Plan integration
        self.plan_id = plan_id
        self.plan = TaskManager.get_plan(plan_id) if plan_id else None

        # Event tracking
        self._tool_events: list[dict] = []

        # Session service (lazy init)
        self._session_service = None

    def _get_session_service(self):
        """Lazy initialization of session service"""
        if self._session_service is None:
            from google.adk.sessions import InMemorySessionService
            self._session_service = InMemorySessionService()
        return self._session_service

    async def execute(
        self, query: str, max_iteration: int = 10, exec_context: ExecutionContext = None
    ) -> AgentResult:
        """Execute query with automatic retry loop"""
        session_service = self._get_session_service()
        session = await session_service.create_session(
            app_name=self.agent.name,
            user_id="default"
        )

        for i in range(max_iteration):
            result = await self._run_once(query, session, exec_context)
            if result.is_complete:
                return result

        return self._handle_max_iteration()

    async def _run_once(
        self, query: str, session, exec_context: ExecutionContext = None
    ) -> AgentResult:
        """Single execution run"""
        from google.adk.runners import Runner
        from google.genai.types import Content, Part

        runner = Runner(
            agent=self.agent,
            session_service=self._get_session_service(),
            app_name=self.agent.name
        )

        events = []
        final_output = ""

        async for event in runner.run_async(
            user_id="default",
            session_id=session.id,
            new_message=Content(parts=[Part(text=query)], role="user")
        ):
            events.append(event)
            self._process_event(event, exec_context)

            if event.is_final_response():
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
                        final_output += part.text

        # Check if complete (no pending tool calls)
        is_complete = True
        for event in events:
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        is_complete = self._has_tool_response(events, part.function_call.name)

        return AgentResult(events=events, output=final_output, is_complete=is_complete)

    def _process_event(self, event, exec_context: ExecutionContext = None) -> None:
        """Process event and track tool calls"""
        if not event.content or not event.content.parts:
            return

        for part in event.content.parts:
            if hasattr(part, 'function_call') and part.function_call:
                call = part.function_call
                pending_calls = exec_context.pending_calls if exec_context else {}
                call_id = f"{call.name}_{len(pending_calls)}"
                pending_calls[call_id] = {
                    "tool": call.name,
                    "args": dict(call.args) if call.args else {},
                    "timestamp": datetime.now().isoformat()
                }
                self._track_tool_event({
                    "type": "call",
                    "name": call.name,
                    "args": call.args
                })
            elif hasattr(part, 'function_response') and part.function_response:
                resp = part.function_response
                self._track_tool_event({
                    "type": "response",
                    "name": resp.name
                })
                # Record to plan if step is active
                self._record_tool_to_plan(resp.name, resp.response, exec_context)

    def _has_tool_response(self, events: list, tool_name: str) -> bool:
        """Check if tool call has a response"""
        for event in events:
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, 'function_response') and part.function_response:
                        if part.function_response.name == tool_name:
                            return True
        return False

    def _handle_max_iteration(self) -> AgentResult:
        """Handle max iteration reached"""
        return AgentResult(
            events=[],
            output="Max iterations reached",
            is_complete=True
        )

    def _track_tool_event(self, event: dict) -> None:
        """Track tool call/response event"""
        self._tool_events.append(event)

    def _record_tool_to_plan(
        self, tool_name: str, result: Any, exec_context: ExecutionContext = None
    ) -> None:
        """Record tool call to plan's step_tool_history"""
        if self.plan is None or exec_context is None:
            return

        step_index = exec_context.step_index
        pending_calls = exec_context.pending_calls

        # Find matching pending call
        matching_call = None
        matching_id = None
        for call_id, call_info in pending_calls.items():
            if call_info["tool"] == tool_name:
                matching_call = call_info
                matching_id = call_id
                break

        if matching_call is None:
            return

        # Record to plan
        self.plan.add_tool_call(
            step_index=step_index,
            tool=matching_call["tool"],
            args=matching_call["args"],
            result=result,
            timestamp=matching_call["timestamp"]
        )

        # Extract and record files
        files = self._extract_files(tool_name, matching_call["args"], result)
        for file_path in files:
            self.plan.add_file(step_index, file_path)

        # Remove from pending
        del pending_calls[matching_id]

    def _extract_files(self, tool_name: str, args: dict, result: Any) -> list[str]:
        """Extract file paths from tool call"""
        files = []

        # Explicit file operations
        if tool_name in ("write_file", "create_directory") and "path" in args:
            files.append(args["path"])

        # Shell redirect patterns: > file.txt or >> file.txt
        if tool_name == "run_command" and "command" in args:
            cmd = args["command"]
            # Match > or >> followed by filename (simple pattern)
            redirect_matches = re.findall(r'>{1,2}\s*([^\s;&|]+)', cmd)
            files.extend(redirect_matches)

        # Python open() patterns
        if tool_name == "run_python" and "code" in args:
            code = args["code"]
            # Match open('file', 'w') or open("file", "w") patterns
            open_matches = re.findall(r"open\s*\(\s*['\"]([^'\"]+)['\"].*['\"][wax]", code)
            files.extend(open_matches)

        return files

    def get_tool_summary(self) -> dict:
        """Get tool usage statistics"""
        calls = [e for e in self._tool_events if e["type"] == "call"]
        responses = [e for e in self._tool_events if e["type"] == "response"]

        return {
            "total_calls": len(calls),
            "total_responses": len(responses),
            "tools_used": list(set(e["name"] for e in calls))
        }

    @staticmethod
    def should_include_aliases(model: Any) -> bool:
        """Check if model supports aliased tool names with special characters.

        Gemini API doesn't support special chars like <|channel|> in function names.
        gpt-oss models may hallucinate these suffixes and need aliases.

        Args:
            model: The LLM model instance

        Returns:
            True if model needs aliased tool names, False otherwise
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
