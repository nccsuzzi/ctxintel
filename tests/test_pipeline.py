"""Tests for the end-to-end ContextIntel pipeline."""

import pytest

from ctxintel.pipeline import ContextIntel
from ctxintel.presets import load_preset


class TestPipeline:
    """End-to-end tests for the ContextIntel pipeline."""

    def test_end_to_end_token_budget(self, sdk, raw_messages):
        """20 messages should result in token_count <= 8000."""
        result = sdk.process(raw_messages)
        assert result.token_count <= 8000

    def test_extracted_memories_positive(self, sdk, raw_messages):
        """Pipeline should extract at least one memory."""
        result = sdk.process(raw_messages)
        assert result.extracted_memories > 0

    def test_compression_ratio_non_negative(self, sdk, raw_messages):
        """compression_ratio should be >= 0.0."""
        result = sdk.process(raw_messages)
        assert result.compression_ratio >= 0.0

    def test_add_message_and_flush(self, tmp_path):
        """add_message + flush workflow should work correctly."""
        sdk = ContextIntel(memory_path=str(tmp_path / "mem.json"))
        sdk.add_message("user", "Hi, my name is Usman.")
        sdk.add_message("assistant", "Hello Usman!")
        sdk.add_message("user", "I prefer Python.")
        sdk.add_message("user", "Build a REST API.")

        result = sdk.flush()
        assert result.token_count > 0
        assert len(result.messages) > 0

    def test_preview_compression(self, sdk, raw_messages):
        """preview_compression should return dict with required keys."""
        preview = sdk.preview_compression(raw_messages)
        assert "total_messages" in preview
        assert "will_keep" in preview
        assert "will_compress" in preview
        assert "estimated_reduction_pct" in preview

    def test_preset_coding_assistant(self, tmp_path):
        """preset='coding_assistant' should load correct config."""
        sdk = ContextIntel(
            preset="coding_assistant",
            memory_path=str(tmp_path / "mem.json"),
        )
        assert sdk.threshold == 0.35
        assert sdk.token_budget == 8000
        assert "coding_language" in sdk.preserve

    def test_reset_memory(self, tmp_path):
        """reset_memory() should clear the memory store."""
        sdk = ContextIntel(memory_path=str(tmp_path / "mem.json"))
        sdk.process([
            {"role": "user", "content": "I prefer Python."},
            {"role": "user", "content": "Build a REST API."},
        ])
        assert len(sdk.memory) > 0
        sdk.reset_memory()
        assert len(sdk.memory) == 0

    def test_invalid_preset_raises(self):
        """An invalid preset name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown preset"):
            load_preset("nonexistent_preset")

    def test_supported_categories(self, sdk):
        """supported_categories should return a non-empty list."""
        cats = sdk.supported_categories()
        assert len(cats) > 0
        assert "task" in cats
