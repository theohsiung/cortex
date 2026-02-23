from app.task.plan import Plan
from app.tools.plan_toolkit import PlanToolkit


class TestPlanToolkit:
    def setup_method(self):
        self.plan = Plan()
        self.toolkit = PlanToolkit(self.plan)

    def test_create_plan(self):
        """Should populate plan with title and steps"""
        result = self.toolkit.create_plan(title="Test Plan", steps=["Step 1", "Step 2"])

        assert self.plan.title == "Test Plan"
        assert self.plan.steps == ["Step 1", "Step 2"]
        assert "Test Plan" in result

    def test_create_plan_with_dependencies(self):
        """Should use provided dependencies"""
        self.toolkit.create_plan(title="Test", steps=["A", "B", "C"], dependencies={2: [0, 1]})

        assert self.plan.dependencies == {2: [0, 1]}

    def test_create_plan_auto_dependencies(self):
        """Should generate sequential dependencies when not provided"""
        self.toolkit.create_plan(title="Test", steps=["A", "B", "C"])

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
        """Should return list of callable functions"""
        functions = self.toolkit.get_tool_functions()

        # Should be a list
        assert isinstance(functions, list)

        # Default: only original tools (no aliases)
        assert len(functions) == 2

        # All should be callable
        assert all(callable(f) for f in functions)

        # Should include create_plan and update_plan
        func_names = [f.__name__ for f in functions]
        assert "create_plan" in func_names
        assert "update_plan" in func_names

    def test_get_tool_functions_with_aliases(self):
        """Should include aliased versions when include_aliases=True"""
        functions = self.toolkit.get_tool_functions(include_aliases=True)

        # Should include original tools and aliased versions (2 + 5*2 = 12)
        assert len(functions) == 12

        # All should be callable
        assert all(callable(f) for f in functions)


class TestPlanToolkitIntents:
    def setup_method(self):
        self.plan = Plan()
        self.toolkit = PlanToolkit(self.plan)

    def test_create_plan_with_intents(self):
        """create_plan should accept and set intents"""
        self.toolkit.create_plan(
            title="Test",
            steps=["Generate code", "Review code"],
            dependencies={1: [0]},
            intents={0: "generate", 1: "review"},
        )
        assert self.plan.step_intents[0] == "generate"
        assert self.plan.step_intents[1] == "review"

    def test_create_plan_without_intents_defaults(self):
        """create_plan without intents should default to 'default'"""
        self.toolkit.create_plan(title="Test", steps=["A", "B"], dependencies={1: [0]})
        assert self.plan.step_intents[0] == "default"
        assert self.plan.step_intents[1] == "default"

    def test_create_plan_with_string_key_intents(self):
        """create_plan should handle string keys from JSON (LLM output)"""
        self.toolkit.create_plan(
            title="Test",
            steps=["Generate code", "Review code"],
            dependencies={1: [0]},
            intents={"0": "generate", "1": "review"},
        )
        assert self.plan.step_intents[0] == "generate"
        assert self.plan.step_intents[1] == "review"

    def test_update_plan_with_intents(self):
        """update_plan should accept and set intents"""
        self.toolkit.create_plan(title="Test", steps=["A", "B"], dependencies={1: [0]})
        self.toolkit.update_plan(
            steps=["Generate code", "Fix bugs", "Review code"],
            dependencies={1: [0], 2: [1]},
            intents={0: "generate", 1: "fix", 2: "review"},
        )
        assert self.plan.step_intents[0] == "generate"
        assert self.plan.step_intents[1] == "fix"
        assert self.plan.step_intents[2] == "review"

    def test_create_plan_schema_includes_intents(self):
        """CREATE_PLAN_SCHEMA should include intents property"""
        schema = PlanToolkit.CREATE_PLAN_SCHEMA
        properties = schema.parameters["properties"]
        assert "intents" in properties
        assert properties["intents"]["type"] == "object"

    def test_update_plan_schema_includes_intents(self):
        """UPDATE_PLAN_SCHEMA should include intents property"""
        schema = PlanToolkit.UPDATE_PLAN_SCHEMA
        properties = schema.parameters["properties"]
        assert "intents" in properties
        assert properties["intents"]["type"] == "object"
