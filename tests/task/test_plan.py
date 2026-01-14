import pytest
from app.task.plan import Plan


class TestPlan:
    def test_create_empty_plan(self):
        """Should create plan with empty defaults"""
        plan = Plan()

        assert plan.title == ""
        assert plan.steps == []
        assert plan.dependencies == {}

    def test_create_plan_with_title_and_steps(self):
        """Should create plan with provided values"""
        plan = Plan(
            title="Test Plan",
            steps=["Step 1", "Step 2", "Step 3"]
        )

        assert plan.title == "Test Plan"
        assert len(plan.steps) == 3
        # Default sequential dependencies
        assert plan.dependencies == {1: [0], 2: [1]}

    def test_create_plan_with_custom_dependencies(self):
        """Should use custom dependencies when provided"""
        plan = Plan(
            title="Test",
            steps=["A", "B", "C"],
            dependencies={2: [0, 1]}  # C depends on both A and B
        )

        assert plan.dependencies == {2: [0, 1]}

    def test_update_plan(self):
        """Should update plan properties"""
        plan = Plan()

        plan.update(
            title="Updated",
            steps=["New Step 1", "New Step 2"],
            dependencies={1: [0]}
        )

        assert plan.title == "Updated"
        assert plan.steps == ["New Step 1", "New Step 2"]

    def test_step_statuses_initialized(self):
        """Should initialize all steps as not_started"""
        plan = Plan(steps=["A", "B", "C"])

        assert plan.step_statuses["A"] == "not_started"
        assert plan.step_statuses["B"] == "not_started"
        assert plan.step_statuses["C"] == "not_started"

    def test_mark_step_status(self):
        """Should update step status"""
        plan = Plan(steps=["A", "B"])

        plan.mark_step(0, step_status="in_progress")

        assert plan.step_statuses["A"] == "in_progress"

    def test_mark_step_with_notes(self):
        """Should store step notes"""
        plan = Plan(steps=["A", "B"])

        plan.mark_step(0, step_status="completed", step_notes="Done successfully")

        assert plan.step_notes["A"] == "Done successfully"

    def test_get_ready_steps_initial(self):
        """Should return first step when no dependencies completed"""
        plan = Plan(steps=["A", "B", "C"])

        ready = plan.get_ready_steps()

        assert ready == [0]  # Only first step is ready

    def test_get_ready_steps_after_completion(self):
        """Should return next steps after dependencies complete"""
        plan = Plan(steps=["A", "B", "C"])
        plan.mark_step(0, step_status="completed")

        ready = plan.get_ready_steps()

        assert ready == [1]  # Second step is now ready

    def test_get_ready_steps_parallel(self):
        """Should return multiple ready steps for parallel execution"""
        plan = Plan(
            steps=["A", "B", "C", "D"],
            dependencies={2: [0], 3: [1]}  # C depends on A, D depends on B
        )

        ready = plan.get_ready_steps()

        assert set(ready) == {0, 1}  # A and B can run in parallel

    def test_get_progress(self):
        """Should return progress statistics"""
        plan = Plan(steps=["A", "B", "C"])
        plan.mark_step(0, step_status="completed")
        plan.mark_step(1, step_status="in_progress")

        progress = plan.get_progress()

        assert progress["total"] == 3
        assert progress["completed"] == 1
        assert progress["in_progress"] == 1
        assert progress["not_started"] == 1

    def test_format_plan(self):
        """Should return formatted string representation"""
        plan = Plan(title="Test", steps=["A", "B"])

        output = plan.format()

        assert "Test" in output
        assert "A" in output
        assert "B" in output
