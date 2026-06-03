"""Token optimizer that fits messages and memories within a token budget."""

import warnings
from typing import List, Optional

from ctxintel.models import ContextResult, Memory, Message
from ctxintel.memory import MemoryStore

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    _count_tokens = lambda text: len(_enc.encode(text))
    _HAS_TIKTOKEN = True
except (ImportError, Exception):
    _HAS_TIKTOKEN = False
    _count_tokens = lambda text: len(text) // 4
    warnings.warn(
        "tiktoken not found. Using len//4 approximation for token counting. "
        "Install with: pip install tiktoken",
        stacklevel=2,
    )


class Optimizer:
    """Fits conversation messages and memories within a token budget.

    Uses tiktoken (cl100k_base) for accurate token counting when available,
    falls back to len//4 approximation otherwise.
    """

    def __init__(self) -> None:
        self._count = _count_tokens

    def optimize(
        self,
        messages: List[Message],
        memories: List[Memory],
        token_budget: int = 8000,
        memory_store: Optional[MemoryStore] = None,
    ) -> ContextResult:
        """Optimize messages and memories to fit within the token budget.

        Args:
            messages: List of Message objects (already ranked and compressed).
            memories: List of Memory objects to include in the result.
            token_budget: Maximum token count for the final output.
            memory_store: Optional MemoryStore for injecting context string.

        Returns:
            A ContextResult with the optimized messages and metadata.
        """
        # Step 1: Count original tokens
        original_tokens = sum(self._count(m.content) for m in messages)
        remaining_budget = token_budget

        # Step 2: Inject memory context if available
        # Guard: memory block must be < 20% of original conversation size
        MAX_MEMORY_RATIO = 0.20
        context_injected_tokens = 0
        
        if memory_store and len(memory_store) > 0:
            ctx_str = memory_store.to_context_string()
            if ctx_str:
                memory_tokens = self._count(ctx_str)
                if memory_tokens <= original_tokens * MAX_MEMORY_RATIO:
                    ctx_msg = Message(
                        role="system", content=ctx_str, importance=1.0, preserved=True
                    )
                    messages = [ctx_msg] + messages
                    remaining_budget -= memory_tokens
                    context_injected_tokens = memory_tokens
                else:
                    # Inject compact top-3 summary instead
                    top_3 = memory_store.get_top(3)
                    if top_3:
                        short = "\n".join(
                            f"{m.key}: {m.value[:40]}" for m in top_3
                        )
                        content = f"[Key context]:\n{short}"
                        ctx_msg = Message(
                            role="system", content=content, importance=1.0, preserved=True
                        )
                        ctx_tokens = self._count(content)
                        messages = [ctx_msg] + messages
                        remaining_budget -= ctx_tokens
                        context_injected_tokens = ctx_tokens

        # Step 3: Separate preserved messages
        preserved = [m for m in messages if m.preserved]
        non_preserved = [m for m in messages if not m.preserved]

        preserved_tokens = sum(self._count(m.content) for m in preserved)
        remaining_budget -= preserved_tokens

        # Step 4: Greedily add non-preserved by importance
        non_preserved.sort(key=lambda m: m.importance, reverse=True)
        selected: List[Message] = []
        for msg in non_preserved:
            t = self._count(msg.content)
            if t <= remaining_budget:
                selected.append(msg)
                remaining_budget -= t

        # Step 5: Restore original order
        final = preserved + selected
        # Use a stable sort by original index (position in the input list)
        msg_order = {id(m): i for i, m in enumerate(messages)}
        final.sort(key=lambda m: msg_order.get(id(m), 0))

        # Step 6: Build result
        used_tokens = sum(self._count(m.content) for m in final)
        ratio = round(1 - used_tokens / max(original_tokens, 1), 3)
        ratio = max(0.0, ratio)

        return ContextResult(
            messages=final,
            memories=memories,
            token_count=used_tokens,
            original_token_count=original_tokens,
            compression_ratio=ratio,
            extracted_memories=len(memories),
            context_injected=context_injected_tokens,
        )
