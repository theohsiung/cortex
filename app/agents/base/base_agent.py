from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

from app.task.task_manager import TaskManager

# Lazy imports to avoid google.adk triggering directory creation on import
if TYPE_CHECKING:
    from google.adk.agents import LlmAgent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai.types import Content, Part


@dataclass
class AgentResult:
    """Result from agent execution"""
    events: list[Any]
    output: str
    is_complete: bool = True


class BaseAgent:
    """Base class for all agents, wraps Google ADK LlmAgent"""

    def __init__(
        self,
        name: str,
        model: Any,
        tool_declarations: list,
        tool_functions: dict,
        instruction: str,
        plan_id: str = None
    ):
        # Store init params for lazy agent creation
        self._name = name
        self._model = model
        self._tool_declarations = tool_declarations
        self._instruction = instruction
        self._agent = None
        self._session_service = None

        # Tool function mapping for execution
        self.tool_functions = tool_functions

        # Plan integration
        self.plan_id = plan_id
        self.plan = TaskManager.get_plan(plan_id) if plan_id else None

        # Event tracking
        self._tool_events: list[dict] = []

    @property
    def agent(self):
        """Lazy initialization of ADK LlmAgent"""
        if self._agent is None:
            from google.adk.agents import LlmAgent
            self._agent = LlmAgent(
                name=self._name,
                model=self._model,
                tools=self._tool_declarations,
                instruction=self._instruction
            )
        return self._agent

    def _get_session_service(self):
        """Lazy initialization of session service"""
        if self._session_service is None:
            from google.adk.sessions import InMemorySessionService
            self._session_service = InMemorySessionService()
        return self._session_service

    async def execute(self, query: str, max_iteration: int = 10) -> AgentResult:
        """Execute query with automatic retry loop"""
        session_service = self._get_session_service()
        session = await session_service.create_session(
            app_name=self.agent.name,
            user_id="default"
        )

        for i in range(max_iteration):
            result = await self._run_once(query, session)
            if result.is_complete:
                return result

        return self._handle_max_iteration()

    async def _run_once(self, query: str, session) -> AgentResult:
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
            self._process_event(event)

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
                        # Has tool call, need to check if responded
                        is_complete = self._has_tool_response(events, part.function_call.name)

        return AgentResult(events=events, output=final_output, is_complete=is_complete)

    def _process_event(self, event) -> None:
        """Process event and track tool calls"""
        if not event.content or not event.content.parts:
            return

        for part in event.content.parts:
            if hasattr(part, 'function_call') and part.function_call:
                self._track_tool_event({
                    "type": "call",
                    "name": part.function_call.name,
                    "args": part.function_call.args
                })
            elif hasattr(part, 'function_response') and part.function_response:
                self._track_tool_event({
                    "type": "response",
                    "name": part.function_response.name
                })

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

    def get_tool_summary(self) -> dict:
        """Get tool usage statistics"""
        calls = [e for e in self._tool_events if e["type"] == "call"]
        responses = [e for e in self._tool_events if e["type"] == "response"]

        return {
            "total_calls": len(calls),
            "total_responses": len(responses),
            "tools_used": list(set(e["name"] for e in calls))
        }
