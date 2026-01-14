import pytest
from app.task.plan import Plan
from app.tools.plan_toolkit import PlanToolkit


class TestPlanToolkit:
    def setup_method(self):
        self.plan = Plan()
        self.toolkit = PlanToolkit(self.plan)

    def test_create_plan(self):
        """Should populate plan with title and steps"""
        result = self.toolkit.create_plan(
            title="Test Plan",
            steps=["Step 1", "Step 2"]
        )

        assert self.plan.title == "Test Plan"
        assert self.plan.steps == ["Step 1", "Step 2"]
        assert "Test Plan" in result

    def test_create_plan_with_dependencies(self):
        """Should use provided dependencies"""
        self.toolkit.create_plan(
            title="Test",
            steps=["A", "B", "C"],
            dependencies={2: [0, 1]}
        )

        assert self.plan.dependencies == {2: [0, 1]}

    def test_create_plan_auto_dependencies(self):
        """Should generate sequential dependencies when not provided"""
        self.toolkit.create_plan(
            title="Test",
            steps=["A", "B", "C"]
        )

        assert self.plan.dependencies == {1: [0], 2: [1]}

    def test_update_plan_title(self):
        """Should update only title when only title provided"""
        self.toolkit.create_plan(title="Original", steps=["A", "B"])

        self.toolkit.update_plan(title="Updated")

        assert self.plan.title == "Updated"
        assert self.plan.steps == ["A", "B"]

    def test_update_plan_steps(self):
        """Should update steps"""
        self.toolkit.create_plan(title="Test", steps=["A"])

        self.toolkit.update_plan(steps=["A", "B", "C"])

        assert self.plan.steps == ["A", "B", "C"]

    def test_get_tool_declarations(self):
        """Should return list of FunctionDeclarations"""
        declarations = self.toolkit.get_tool_declarations()

        assert len(declarations) == 2
        names = [d.name for d in declarations]
        assert "create_plan" in names
        assert "update_plan" in names

    def test_get_tool_functions(self):
        """Should return dict mapping names to functions"""
        functions = self.toolkit.get_tool_functions()

        assert "create_plan" in functions
        assert "update_plan" in functions
        assert callable(functions["create_plan"])
