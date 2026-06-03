"""Tests for the MemoryStore module."""

from ctxintel.models import Memory
from ctxintel.memory import MemoryStore


class TestMemoryStore:
    """Tests for persistent memory storage."""

    def test_save_and_reload(self, tmp_path):
        """Memories should persist to JSON and reload correctly."""
        path = str(tmp_path / "mem.json")
        store = MemoryStore(path)
        store.save([Memory(key="task", value="build API", importance=0.8)])
        assert len(store) == 1

        # Create a new store from the same path — should reload
        store2 = MemoryStore(path)
        assert len(store2) == 1
        assert store2.get_by_key("task")[0].value == "build API"

    def test_frequency_increment(self, tmp_path):
        """Saving the same memory twice should increment frequency to 2."""
        path = str(tmp_path / "mem.json")
        store = MemoryStore(path)
        mem = Memory(key="task", value="build API", importance=0.5)
        store.save([mem])
        store.save([Memory(key="task", value="build API", importance=0.5)])
        tasks = store.get_by_key("task")
        assert tasks[0].frequency == 2

    def test_get_top(self, tmp_path):
        """get_top(3) should return exactly 3 highest importance memories."""
        path = str(tmp_path / "mem.json")
        store = MemoryStore(path)
        store.save([
            Memory(key="a", value="v1", importance=0.9),
            Memory(key="b", value="v2", importance=0.5),
            Memory(key="c", value="v3", importance=0.7),
            Memory(key="d", value="v4", importance=0.3),
            Memory(key="e", value="v5", importance=0.8),
        ])
        top = store.get_top(3)
        assert len(top) == 3
        assert top[0].importance >= top[1].importance >= top[2].importance

    def test_get_by_key(self, tmp_path):
        """get_by_key should return only memories with the matching key."""
        path = str(tmp_path / "mem.json")
        store = MemoryStore(path)
        store.save([
            Memory(key="task", value="build API", importance=0.8),
            Memory(key="pref", value="Python", importance=0.7),
            Memory(key="task", value="add auth", importance=0.6),
        ])
        tasks = store.get_by_key("task")
        assert len(tasks) == 2
        assert all(m.key == "task" for m in tasks)

    def test_to_context_string(self, tmp_path):
        """to_context_string should contain the memory key and value."""
        path = str(tmp_path / "mem.json")
        store = MemoryStore(path)
        store.save([Memory(key="task", value="build API", importance=0.8)])
        ctx = store.to_context_string()
        assert "task" in ctx
        assert "build API" in ctx

    def test_clear(self, tmp_path):
        """clear() should result in len(store) == 0."""
        path = str(tmp_path / "mem.json")
        store = MemoryStore(path)
        store.save([Memory(key="task", value="build API", importance=0.8)])
        assert len(store) > 0
        store.clear()
        assert len(store) == 0

    def test_context_string_empty_when_low_importance(self, tmp_path):
        """Memories below 0.4 importance should not appear in context string."""
        path = str(tmp_path / "mem.json")
        store = MemoryStore(path)
        store.save([Memory(key="task", value="minor thing", importance=0.1)])
        ctx = store.to_context_string()
        assert ctx == ""
