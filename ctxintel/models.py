"""Data models for ctxintel pipeline."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Message:
    """A single conversation message with importance metadata.

    Attributes:
        role: The role of the message sender ("user", "assistant", or "system").
        content: The text content of the message.
        importance: Computed importance score between 0.0 and 1.0.
        preserved: If True, this message will never be compressed or removed.
    """

    role: str
    content: str
    importance: float = 0.0
    preserved: bool = False


@dataclass
class Memory:
    """A single extracted fact or piece of context from a conversation.

    Attributes:
        key: Category of the memory (e.g. "user_preference", "task", "constraint").
        value: The extracted value or fact.
        importance: Scored 0.0–1.0 based on recency, emphasis, and role.
        frequency: How many times this fact has appeared across conversations.
        source_role: Who said it — "user", "assistant", or "system".
        timestamp_index: Position in the conversation where this was extracted.
    """

    key: str
    value: str
    importance: float = 0.0
    frequency: int = 1
    source_role: str = "user"
    timestamp_index: int = 0


@dataclass
class ContextResult:
    """Result of the full ctxintel pipeline.

    Attributes:
        messages: The final list of optimized messages.
        memories: The list of extracted memories included in context.
        token_count: Token count of the final output.
        original_token_count: Token count before processing.
        compression_ratio: How much the context was reduced (0.0–1.0).
        extracted_memories: Total number of memories extracted.
    """

    messages: List[Message] = field(default_factory=list)
    memories: List[Memory] = field(default_factory=list)
    token_count: int = 0
    original_token_count: int = 0
    compression_ratio: float = 0.0
    extracted_memories: int = 0
    tokens_saved: int = 0
    context_injected: int = 0
