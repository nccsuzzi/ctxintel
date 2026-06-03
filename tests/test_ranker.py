"""Tests for the Ranker module."""

from ctxintel.models import Message
from ctxintel.ranker import Ranker


class TestRanker:
    """Tests for message importance ranking."""

    def setup_method(self):
        self.ranker = Ranker()

    def test_system_messages_always_score_one(self):
        """System messages should always receive importance 1.0."""
        msgs = [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="Hello there!"),
        ]
        ranked = self.ranker.rank(msgs)
        assert ranked[0].importance == 1.0

    def test_recent_message_scores_higher(self):
        """Most recent message should score higher than the first message."""
        msgs = [
            Message(role="user", content="This is my first message about Python programming."),
            Message(role="user", content="This is a middle message about databases and storage."),
            Message(role="user", content="This is the latest message about deployment strategy."),
        ]
        ranked = self.ranker.rank(msgs)
        assert ranked[-1].importance > ranked[0].importance

    def test_short_message_penalty(self):
        """Messages under 15 chars should score lower than longer messages."""
        msgs = [
            Message(role="user", content="ok"),
            Message(role="user", content="I need to build a complete REST API with authentication."),
        ]
        ranked = self.ranker.rank(msgs)
        assert ranked[0].importance < ranked[1].importance

    def test_repeated_content_scores_lower(self):
        """Repeated content should score lower on uniqueness."""
        msgs = [
            Message(role="user", content="I want to build a Python REST API with FastAPI."),
            Message(role="user", content="I want to build a Python REST API with FastAPI."),
            Message(role="user", content="Deploy everything on AWS ECS containers."),
        ]
        ranked = self.ranker.rank(msgs)
        # The repeated message should score lower than the unique one
        assert ranked[1].importance <= ranked[2].importance

    def test_all_scores_in_range(self, sample_messages):
        """All scores must be between 0.0 and 1.0."""
        ranked = self.ranker.rank(sample_messages)
        for msg in ranked:
            assert 0.0 <= msg.importance <= 1.0

    def test_empty_list_returns_empty(self):
        """Ranking an empty list should return an empty list."""
        assert self.ranker.rank([]) == []

    def test_single_message(self):
        """A single message should be rankable without error."""
        msgs = [Message(role="user", content="Hello world")]
        ranked = self.ranker.rank(msgs)
        assert len(ranked) == 1
        assert 0.0 <= ranked[0].importance <= 1.0
