import pytest
from app.task.plan import Plan
from app.tools.act_toolkit import ActToolkit


class TestActToolkit:
    def setup_method(self):
        self.plan = Plan(title="Test", steps=["A", "B", "C"])
        self.toolkit = ActToolkit(self.plan)

    def test_mark_step_in_progress(self):
        """Should mark step as in_progress"""
        result = self.toolkit.mark_step(step_index=0, status="in_progress")

        assert self.plan.step_statuses["A"] == "in_progress"
        assert "0" in result
        assert "in_progress" in result

    def test_mark_step_completed(self):
        """Should mark step as completed"""
        self.toolkit.mark_step(step_index=0, status="completed")

        assert self.plan.step_statuses["A"] == "completed"

    def test_mark_step_with_notes(self):
        """Should store notes with step"""
        self.toolkit.mark_step(
            step_index=0,
            status="completed",
            notes="Successfully completed"
        )

        assert self.plan.step_notes["A"] == "Successfully completed"

    def test_mark_step_blocked(self):
        """Should mark step as blocked"""
        self.toolkit.mark_step(step_index=1, status="blocked", notes="Dependency failed")

        assert self.plan.step_statuses["B"] == "blocked"

    def test_mark_step_invalid_index(self):
        """Should raise error for invalid step index"""
        with pytest.raises(ValueError):
            self.toolkit.mark_step(step_index=99, status="completed")

    def test_get_tool_declarations(self):
        """Should return list with mark_step declaration"""
        declarations = self.toolkit.get_tool_declarations()

        assert len(declarations) == 1
        assert declarations[0].name == "mark_step"

    def test_get_tool_functions(self):
        """Should return dict with mark_step function"""
        functions = self.toolkit.get_tool_functions()

        assert "mark_step" in functions
        assert callable(functions["mark_step"])
