"""Main pipeline orchestrator for ctxintel."""

from typing import Dict, List, Optional

from ctxintel.models import ContextResult, Message
from ctxintel.ranker import Ranker
from ctxintel.extractor import Extractor
from ctxintel.memory import MemoryStore
from ctxintel.compressor import Compressor
from ctxintel.optimizer import Optimizer
from ctxintel.presets import load_preset


class ContextIntel:
    """Context Operating System for LLM Applications.

    Orchestrates the full 6-stage pipeline: ranking, extraction, memory,
    compression, optimization, and final context assembly.

    Args:
        preset: Name of a built-in preset (e.g. "coding_assistant").
        preserve: List of memory categories to always preserve.
        threshold: Importance threshold for compression (0.0–1.0).
        token_budget: Maximum token count for the final context.
        memory_path: File path for persistent memory JSON storage.
    """

    def __init__(
        self,
        preset: Optional[str] = None,
        preserve: Optional[List[str]] = None,
        threshold: float = 0.4,
        token_budget: int = 8000,
        memory_path: str = ".ctxintel_memory.json",
    ) -> None:
        # Load preset defaults if specified
        if preset is not None:
            config = load_preset(preset)
            preserve = preserve or config.get("preserve", [])
            threshold = threshold if threshold != 0.4 else config.get("threshold", 0.4)
            token_budget = token_budget if token_budget != 8000 else config.get("token_budget", 8000)

        self.preserve: List[str] = preserve or []
        self.threshold: float = threshold
        self.token_budget: int = token_budget

        self.ranker = Ranker()
        self.extractor = Extractor()
        self.memory = MemoryStore(memory_path)
        self.compressor = Compressor()
        self.optimizer = Optimizer()
        self._buffer: List[dict] = []

    def process(
        self,
        raw_messages: List[dict],
        preserve: Optional[List[str]] = None,
        token_budget: Optional[int] = None,
    ) -> ContextResult:
        """Run the full ctxintel pipeline on a list of raw messages.

        Args:
            raw_messages: List of dicts with "role" and "content" keys.
            preserve: Override the default preserve categories.
            token_budget: Override the default token budget.

        Returns:
            A ContextResult with optimized messages and metadata.
        """
        preserve = preserve or self.preserve
        token_budget = token_budget or self.token_budget

        # Convert dicts to Message objects
        messages = [Message(role=m["role"], content=m["content"]) for m in raw_messages]

        # 1. Rank
        messages = self.ranker.rank(messages)

        # Capture original token count BEFORE compression (using same counter as optimizer)
        original_token_count = sum(self.optimizer._count(m.content) for m in messages)

        # 2. Extract
        memories = self.extractor.extract(messages, preserve)

        # 3. Save to memory store
        self.memory.save(memories)

        # 4. Compress
        messages = self.compressor.compress(messages, preserve, self.threshold)

        # 5. Optimize
        result = self.optimizer.optimize(
            messages, self.memory.get_top(20), token_budget, self.memory
        )

        # Calculate exactly how many tokens were saved (removed via compression)
        result.tokens_saved = max(0, original_token_count + result.context_injected - result.token_count)
        
        # Override with pre-compression token count for accurate ratio
        result.original_token_count = original_token_count
        
        # True compression ratio (how much of the original text was compressed away)
        ratio = result.tokens_saved / max(original_token_count, 1)
        result.compression_ratio = min(1.0, max(0.0, round(ratio, 3)))

        return result

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the internal buffer for later flushing.

        Args:
            role: The role of the sender ("user", "assistant", "system").
            content: The text content of the message.
        """
        self._buffer.append({"role": role, "content": content})

    def flush(self) -> ContextResult:
        """Process all buffered messages and clear the buffer.

        Returns:
            A ContextResult from processing all buffered messages.
        """
        result = self.process(self._buffer)
        self._buffer = []
        return result

    def memory_summary(self) -> str:
        """Return a formatted string of stored memories for inspection.

        Returns:
            Formatted memory context string.
        """
        return self.memory.to_context_string()

    def reset_memory(self) -> None:
        """Clear all stored memories and delete the persistence file."""
        self.memory.clear()

    def preview_compression(self, raw_messages: List[dict]) -> Dict:
        """Preview compression statistics without actually compressing.

        Args:
            raw_messages: List of dicts with "role" and "content" keys.

        Returns:
            Dict with total_messages, will_keep, will_compress,
            estimated_reduction_pct.
        """
        messages = [Message(role=m["role"], content=m["content"]) for m in raw_messages]
        messages = self.ranker.rank(messages)
        return self.compressor.estimate_reduction(messages, self.threshold)

    def supported_categories(self) -> List[str]:
        """Return all memory categories available from patterns.yaml.

        Returns:
            List of category name strings.
        """
        return self.extractor.get_supported_categories()
