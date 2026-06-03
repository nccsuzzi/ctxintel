"""Persistent memory store with JSON-backed storage and frequency tracking."""

import json
import os
import string
import difflib
from pathlib import Path
from typing import Dict, List, Optional

from ctxintel.models import Memory


def _normalize_value(val: str) -> str:
    """Normalize a value for deduplication: lowercase, strip, remove punctuation."""
    val = val.lower().strip()
    val = val.translate(str.maketrans("", "", string.punctuation))
    return val.strip()


class MemoryStore:
    """Persistent memory store backed by a JSON file.

    Tracks extracted facts across conversations with frequency counting,
    importance recalculation, and formatted context output for LLM injection.

    Args:
        path: File path for the JSON persistence file.
    """

    def __init__(self, path: str = ".ctxintel_memory.json") -> None:
        self._path = path
        self._memories: List[Memory] = []
        self._load()

    def save(self, memories: List[Memory]) -> None:
        """Merge incoming memories into the store, updating frequency and importance.

        For each incoming memory:
        - If the same key + normalized value already exists, increment frequency,
          recalculate importance, and update the timestamp.
        - Otherwise, append as a new memory.

        Args:
            memories: List of Memory objects to save.
        """
        for mem in memories:
            existing = self._find_existing(mem.key, mem.value)
            if existing is not None:
                existing.frequency += 1
                existing.timestamp_index = max(
                    existing.timestamp_index, mem.timestamp_index
                )
                # Recalculate importance with frequency boost
                freq_boost = min(existing.frequency * 0.1, 0.3)
                existing.importance = min(
                    1.0,
                    round(max(existing.importance, mem.importance) + freq_boost, 3),
                )
            else:
                self._memories.append(mem)

        self._write()

    def get_by_key(self, key: str) -> List[Memory]:
        """Return all memories matching the given key, sorted by importance descending.

        Args:
            key: The memory category key to filter by.

        Returns:
            List of Memory objects matching the key.
        """
        return sorted(
            [m for m in self._memories if m.key == key],
            key=lambda m: m.importance,
            reverse=True,
        )

    def get_top(self, n: int, key: Optional[str] = None) -> List[Memory]:
        """Return the top n memories by importance.

        Args:
            n: Maximum number of memories to return.
            key: If given, filter by this key before ranking.

        Returns:
            List of up to n Memory objects, sorted by importance descending.
        """
        pool = self._memories
        if key:
            pool = [m for m in pool if m.key == key]

        return sorted(pool, key=lambda m: m.importance, reverse=True)[:n]

    def to_context_string(self) -> str:
        """Format stored memories as a readable block for LLM context injection.

        Returns:
            A formatted string containing the top memories with importance
            and frequency metadata. Only includes memories with importance >= 0.5.
            Returns an empty string if no qualifying memories exist.
        """
        top = self.get_top(10)
        top = [m for m in top if m.importance >= 0.5]

        if not top:
            return ""

        lines = ["[Known context]"]
        for m in top:
            value = m.value[:40]
            lines.append(f"- {m.key}: {value}")

        return "\n".join(lines)

    def clear(self) -> None:
        """Wipe all in-memory data and delete the JSON file if it exists."""
        self._memories = []
        if os.path.exists(self._path):
            os.remove(self._path)

    def __len__(self) -> int:
        """Return the count of stored memories."""
        return len(self._memories)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_existing(self, key: str, value: str) -> Optional[Memory]:
        """Find an existing memory with fuzzy matching."""
        norm = _normalize_value(value)
        best_match = None
        best_ratio = 0.0

        for mem in self._memories:
            if mem.key != key:
                continue

            exist_norm = _normalize_value(mem.value)
            
            # Substring match
            if (norm in exist_norm) or (exist_norm in norm):
                # Update to the longer, more descriptive value if needed
                if len(value) > len(mem.value):
                    mem.value = value
                return mem
                
            # Fuzzy match
            ratio = difflib.SequenceMatcher(None, norm, exist_norm).ratio()
            if ratio > 0.65 and ratio > best_ratio:
                best_ratio = ratio
                best_match = mem

        if best_match:
            if len(value) > len(best_match.value):
                best_match.value = value
            return best_match
            
        return None

    def _load(self) -> None:
        """Load memories from the JSON file if it exists."""
        if not os.path.exists(self._path):
            return

        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self._memories = [
                Memory(
                    key=item["key"],
                    value=item["value"],
                    importance=item.get("importance", 0.0),
                    frequency=item.get("frequency", 1),
                    source_role=item.get("source_role", "user"),
                    timestamp_index=item.get("timestamp_index", 0),
                )
                for item in data
            ]
        except (json.JSONDecodeError, KeyError, TypeError):
            self._memories = []

    def _write(self) -> None:
        """Write all memories to the JSON file."""
        data = [
            {
                "key": m.key,
                "value": m.value,
                "importance": m.importance,
                "frequency": m.frequency,
                "source_role": m.source_role,
                "timestamp_index": m.timestamp_index,
            }
            for m in self._memories
        ]

        # Ensure parent directory exists
        parent = os.path.dirname(self._path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
