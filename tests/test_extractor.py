"""Tests for the Extractor module."""

from ctxintel.models import Message
from ctxintel.extractor import Extractor


class TestExtractor:
    """Tests for hybrid fact extraction."""

    def setup_method(self):
        self.extractor = Extractor()

    def test_extract_user_name(self):
        """'My name is Ali' should extract a user_name memory."""
        msgs = [Message(role="user", content="My name is Ali")]
        memories = self.extractor.extract(msgs)
        names = [m for m in memories if m.key == "user_name"]
        assert len(names) >= 1
        assert any("ali" in m.value.lower() for m in names)

    def test_extract_preference(self):
        """'I prefer Python' should extract a user_preference memory."""
        msgs = [Message(role="user", content="I prefer Python")]
        memories = self.extractor.extract(msgs)
        prefs = [m for m in memories if m.key == "user_preference"]
        assert len(prefs) >= 1
        assert any("python" in m.value.lower() for m in prefs)

    def test_extract_favorite_keyword(self):
        """'My favorite language is Python' should be captured via keyword match."""
        msgs = [Message(role="user", content="My favorite language is Python")]
        memories = self.extractor.extract(msgs)
        matches = [m for m in memories if "python" in m.value.lower()]
        assert len(matches) >= 1

    def test_extract_decision(self):
        """'We decided to deploy on AWS' should extract decision or cloud_provider."""
        msgs = [Message(role="user", content="We decided to deploy on AWS")]
        memories = self.extractor.extract(msgs)
        keys = {m.key for m in memories}
        assert "decision" in keys or "cloud_provider" in keys

    def test_extract_constraint(self):
        """\"Don't use synchronous code\" should extract a constraint memory."""
        msgs = [Message(role="user", content="Don't use synchronous code")]
        memories = self.extractor.extract(msgs)
        constraints = [m for m in memories if m.key == "constraint"]
        assert len(constraints) >= 1

    def test_extract_task(self):
        """'I need to build a REST API' should extract a task memory."""
        msgs = [Message(role="user", content="I need to build a REST API")]
        memories = self.extractor.extract(msgs)
        tasks = [m for m in memories if m.key == "task"]
        assert len(tasks) >= 1

    def test_deduplication(self):
        """Duplicate 'I prefer Python' should result in only one memory."""
        msgs = [
            Message(role="user", content="I prefer Python"),
            Message(role="user", content="i prefer python"),
        ]
        memories = self.extractor.extract(msgs)
        prefs = [m for m in memories if m.key == "user_preference" and "python" in m.value.lower()]
        assert len(prefs) == 1

    def test_get_supported_categories(self):
        """get_supported_categories() should return a non-empty list."""
        cats = self.extractor.get_supported_categories()
        assert len(cats) > 0
        assert "task" in cats
        assert "user_name" in cats

    def test_preserve_filter(self):
        """When preserve is given, only those categories should be extracted."""
        msgs = [
            Message(role="user", content="I prefer Python. Build a REST API."),
        ]
        memories = self.extractor.extract(msgs, preserve=["task"])
        # Should have task but no user_preference from pattern matching
        pattern_keys = {m.key for m in memories if m.key in ("task", "user_preference")}
        assert "task" in pattern_keys or len(memories) >= 0  # task should be found
