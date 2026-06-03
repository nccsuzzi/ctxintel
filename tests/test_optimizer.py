"""Tests for the Optimizer module."""

from ctxintel.models import Memory, Message
from ctxintel.optimizer import Optimizer


class TestOptimizer:
    """Tests for token budget optimization."""

    def setup_method(self):
        self.optimizer = Optimizer()

    def test_output_within_budget(self):
        """Output token_count should not exceed token_budget."""
        msgs = [
            Message(role="user", content="Build a REST API " * 50, importance=0.9),
            Message(role="user", content="Deploy on AWS " * 50, importance=0.7),
            Message(role="user", content="Use FastAPI " * 50, importance=0.5),
        ]
        result = self.optimizer.optimize(msgs, [], token_budget=200)
        assert result.token_count <= 200

    def test_preserved_messages_always_included(self):
        """Preserved messages should always appear in output."""
        msgs = [
            Message(role="system", content="System prompt.", importance=1.0, preserved=True),
            Message(role="user", content="Low importance filler.", importance=0.1),
            Message(role="user", content="Build REST API.", importance=0.8),
        ]
        result = self.optimizer.optimize(msgs, [], token_budget=8000)
        contents = [m.content for m in result.messages]
        assert "System prompt." in contents

    def test_compression_ratio_in_range(self):
        """compression_ratio should be between 0.0 and 1.0."""
        msgs = [
            Message(role="user", content="Build a REST API with FastAPI.", importance=0.9),
            Message(role="user", content="ok", importance=0.1),
        ]
        result = self.optimizer.optimize(msgs, [], token_budget=8000)
        assert 0.0 <= result.compression_ratio <= 1.0

    def test_original_gte_final(self):
        """original_token_count should be >= token_count."""
        msgs = [
            Message(role="user", content="Message one about Python.", importance=0.9),
            Message(role="user", content="Message two about AWS.", importance=0.8),
        ]
        result = self.optimizer.optimize(msgs, [], token_budget=8000)
        assert result.original_token_count >= result.token_count

    def test_memories_in_result(self):
        """Memories passed to optimize should appear in the result."""
        msgs = [Message(role="user", content="Hello", importance=0.5)]
        mems = [Memory(key="task", value="build API", importance=0.8)]
        result = self.optimizer.optimize(msgs, mems, token_budget=8000)
        assert result.extracted_memories == 1
        assert result.memories == mems
