"""Global plan manager using singleton pattern via class methods."""

from __future__ import annotations

from threading import Lock
from typing import Any


class TaskManager:
    """Global Plan manager (singleton pattern via class methods)."""

    _lock = Lock()
    _plans: dict[str, Any] = {}

    @classmethod
    def set_plan(cls, plan_id: str, plan: Any) -> None:
        """Register a plan with the given ID."""
        with cls._lock:
            cls._plans[plan_id] = plan

    @classmethod
    def get_plan(cls, plan_id: str) -> Any | None:
        """Retrieve a plan by ID, returns None if not found."""
        with cls._lock:
            return cls._plans.get(plan_id)

    @classmethod
    def remove_plan(cls, plan_id: str) -> None:
        """Remove a plan by ID, no-op if not found."""
        with cls._lock:
            cls._plans.pop(plan_id, None)
