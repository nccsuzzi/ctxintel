"""Tests for the Compressor module."""

from ctxintel.models import Message
from ctxintel.compressor import Compressor


class TestCompressor:
    """Tests for message compression."""

    def setup_method(self):
        self.compressor = Compressor()

    def test_low_importance_messages_compressed(self):
        """Messages below threshold should get compressed into a summary."""
        msgs = [
            Message(role="user", content="ok", importance=0.1),
            Message(role="user", content="cool", importance=0.1),
            Message(role="user", content="hmm sure", importance=0.15),
            Message(role="user", content="I need to build a REST API.", importance=0.8),
        ]
        result = self.compressor.compress(msgs, threshold=0.4)
        # Should have fewer messages than original
        assert len(result) < len(msgs)

    def test_preserved_messages_never_removed(self):
        """Messages with preserved=True should never be removed."""
        msgs = [
            Message(role="system", content="System prompt.", importance=0.1, preserved=True),
            Message(role="user", content="ok", importance=0.1),
            Message(role="user", content="Build API.", importance=0.8),
        ]
        result = self.compressor.compress(msgs, threshold=0.4)
        system_msgs = [m for m in result if m.content == "System prompt."]
        assert len(system_msgs) == 1

    def test_summary_message_in_output(self):
        """Output should contain a message with '[Earlier context summary]' when filler is substantial."""
        msgs = [
            Message(role="user", content="I was thinking about the database schema and how we should structure the tables for user management. We need to consider the relationships between users, roles, and permissions.", importance=0.1),
            Message(role="user", content="Also, I was reading about different authentication strategies and the pros and cons of JWT versus session-based auth. There are many tradeoffs to consider for our specific use case.", importance=0.1),
            Message(role="user", content="Another thing I noticed is that the caching layer needs to handle invalidation properly. We should look into write-through versus write-behind caching strategies.", importance=0.1),
            Message(role="user", content="Important task message.", importance=0.9),
        ]
        result = self.compressor.compress(msgs, threshold=0.4)
        summaries = [m for m in result if "[Earlier context summary]" in m.content]
        assert len(summaries) == 1

    def test_all_high_importance_no_summary(self):
        """If all messages are high importance, no summary should be added."""
        msgs = [
            Message(role="user", content="Build REST API.", importance=0.9),
            Message(role="user", content="Deploy on AWS.", importance=0.8),
            Message(role="user", content="Use FastAPI.", importance=0.7),
        ]
        result = self.compressor.compress(msgs, threshold=0.4)
        summaries = [m for m in result if "[Earlier context summary]" in m.content]
        assert len(summaries) == 0
        assert len(result) == 3

    def test_estimate_reduction(self):
        """estimate_reduction should return correct counts."""
        msgs = [
            Message(role="user", content="ok", importance=0.1),
            Message(role="user", content="Build API.", importance=0.8),
            Message(role="user", content="cool", importance=0.2),
        ]
        est = self.compressor.estimate_reduction(msgs, threshold=0.4)
        assert est["total_messages"] == 3
        assert est["will_keep"] == 1
        assert est["will_compress"] == 2
        assert "estimated_reduction_pct" in est
