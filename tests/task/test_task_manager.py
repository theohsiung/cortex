from app.task.task_manager import TaskManager


class TestTaskManager:
    def setup_method(self):
        """Clear TaskManager state before each test"""
        TaskManager._plans.clear()

    def test_set_and_get_plan(self):
        """Should store and retrieve a plan by ID"""
        plan = {"title": "Test Plan"}
        TaskManager.set_plan("plan_1", plan)

        result = TaskManager.get_plan("plan_1")

        assert result == plan

    def test_get_nonexistent_plan_returns_none(self):
        """Should return None for unknown plan ID"""
        result = TaskManager.get_plan("nonexistent")

        assert result is None

    def test_remove_plan(self):
        """Should remove a plan by ID"""
        plan = {"title": "Test Plan"}
        TaskManager.set_plan("plan_1", plan)

        TaskManager.remove_plan("plan_1")

        assert TaskManager.get_plan("plan_1") is None

    def test_remove_nonexistent_plan_no_error(self):
        """Should not raise error when removing nonexistent plan"""
        TaskManager.remove_plan("nonexistent")  # Should not raise

    def test_thread_safety(self):
        """Should handle concurrent access safely"""
        import threading

        def set_plans():
            for i in range(100):
                TaskManager.set_plan(f"plan_{i}", {"id": i})

        threads = [threading.Thread(target=set_plans) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have stored plans without errors
        assert TaskManager.get_plan("plan_0") is not None
