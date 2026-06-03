"""Hybrid fact extraction using data-driven patterns, spaCy NER, and dependency parsing."""

import difflib
import re
import string
import warnings
from typing import List, Optional, Set

import yaml

from ctxintel.models import Memory, Message

from importlib import resources as importlib_resources


def _load_patterns_config() -> dict:
    """Load the patterns.yaml configuration file using importlib.resources."""
    data_path = importlib_resources.files("ctxintel") / "data" / "patterns.yaml"
    if not data_path.is_file():
        raise FileNotFoundError(
            "patterns.yaml not found. Expected at ctxintel/data/patterns.yaml. "
            "If you installed via pip, reinstall the package. "
            "If developing locally, ensure the file exists in the ctxintel/data/ directory."
        )
    raw = data_path.read_text(encoding="utf-8")
    return yaml.safe_load(raw)


def _normalize_value(val: str) -> str:
    """Normalize a value for deduplication: lowercase, strip, remove punctuation."""
    val = val.lower().strip()
    val = val.translate(str.maketrans("", "", string.punctuation))
    return val.strip()


def _try_load_spacy():
    """Attempt to load the spaCy English model. Returns None if unavailable."""
    try:
        import spacy

        nlp = spacy.load("en_core_web_sm")
        return nlp
    except (ImportError, OSError):
        warnings.warn(
            "spaCy or en_core_web_sm model not found. "
            "NER and dependency parsing will be skipped. "
            "Install with: pip install spacy && python -m spacy download en_core_web_sm",
            stacklevel=2,
        )
        return None


class Extractor:
    """Hybrid fact extractor combining pattern matching, spaCy NER, and dependency parsing.

    Extracts structured Memory objects from conversation messages using three layers:
    1. Data-driven pattern matching from patterns.yaml
    2. Named Entity Recognition via spaCy
    3. Dependency parsing for action intents

    Gracefully degrades if spaCy is unavailable — pattern matching always works.
    """

    def __init__(self) -> None:
        self._config = _load_patterns_config()
        self._categories = self._config.get("categories", {})
        self._entity_map = self._config.get("spacy_entity_map", {})
        self._dep_verbs = set(self._config.get("dependency_verbs", []))
        self._emphasis = self._config.get("emphasis_signals", {})
        self.nlp = _try_load_spacy()

        # Pre-compile regex patterns for each category
        self._compiled_patterns = {}
        for cat_name, cat_data in self._categories.items():
            compiled = []
            for pat in cat_data.get("patterns", []):
                # Convert {X} placeholder to a capturing group
                escaped = re.escape(pat)
                placeholder = re.escape("{X}")

                if cat_name == "task":
                    # Task: greedy capture up to ~5 words, we strip trailing prepositions later
                    capture = r"([\w\s]{2,40})"
                else:
                    capture = r"([\w\s\+\.\#]{2,60})"

                regex_str = escaped.replace(placeholder, capture)

                try:
                    compiled.append(re.compile(regex_str, re.IGNORECASE))
                except re.error:
                    continue
            self._compiled_patterns[cat_name] = compiled

    def extract(
        self,
        messages: List[Message],
        preserve: Optional[List[str]] = None,
    ) -> List[Memory]:
        """Extract Memory objects from a list of messages.

        Args:
            messages: List of Message objects to extract facts from.
            preserve: Optional list of category names to restrict extraction to.
                      If None, all categories are used.

        Returns:
            Deduplicated list of Memory objects extracted from the messages.
        """
        all_memories: List[Memory] = []
        total = len(messages)

        for idx, msg in enumerate(messages):
            text = msg.content

            # Layer 1: Pattern matching
            pattern_memories = self._extract_patterns(
                text, idx, msg.role, preserve
            )
            all_memories.extend(pattern_memories)

            # Layer 2: spaCy NER
            ner_memories = self._extract_ner(text, idx, msg.role)
            all_memories.extend(ner_memories)

            # Layer 3: Dependency parsing
            dep_memories = self._extract_dependencies(text, idx, msg.role)
            all_memories.extend(dep_memories)

        # Score all memories
        self._score_memories(all_memories, messages, total)

        # Deduplicate
        all_memories = self._deduplicate(all_memories)

        return all_memories

    def get_supported_categories(self) -> List[str]:
        """Return all category names available in patterns.yaml.

        Returns:
            List of category name strings.
        """
        return list(self._categories.keys())

    # ------------------------------------------------------------------
    # Hardcoded supplemental patterns — always run regardless of preserve
    # ------------------------------------------------------------------

    _HARDCODED_PATTERNS = [
        # user_name
        (r"(?:I'?m|my name is|call me|i go by)\s+([A-Z][a-z]+)", "user_name"),
        # constraint
        (r"(?:must be|everything must be)\s+([\w\s]{2,30})", "constraint"),
        (r"[Nn]o\s+(\w{3,20})\s+code", "constraint"),
        (r"[Dd]on'?t\s+use\s+([\w\s]{2,30})", "constraint"),
        # decision
        (r"decided\s+(?:to\s+use\s+)?([A-Z][\w]+)", "decision"),
        (r"we will use\s+([A-Z][\w]+)", "decision"),
    ]

    def _extract_patterns(
        self,
        text: str,
        idx: int,
        role: str,
        preserve: Optional[List[str]],
    ) -> List[Memory]:
        """Extract memories from text using YAML-defined patterns."""
        memories: List[Memory] = []
        lower_text = text.lower()

        # --- Hardcoded patterns (always run, no preserve filter) ---
        for pattern, key in self._HARDCODED_PATTERNS:
            match = re.search(pattern, text)
            if match:
                captured = match.group(1).strip().rstrip(".,!?")
                if captured:
                    memories.append(
                        Memory(
                            key=key,
                            value=captured,
                            source_role=role,
                            timestamp_index=idx,
                        )
                    )

        # --- Keyword-based matching (always run for keyword categories) ---
        for cat_name, cat_data in self._categories.items():
            keywords = cat_data.get("keywords")
            if not keywords:
                continue
            for kw in keywords:
                if re.search(r"\b" + re.escape(kw.lower()) + r"\b", lower_text):
                    memories.append(
                        Memory(
                            key=cat_name,
                            value=kw.lower(),
                            source_role=role,
                            timestamp_index=idx,
                        )
                    )

        # --- YAML pattern-based matching (respects preserve filter) ---
        for cat_name, cat_data in self._categories.items():
            # If preserve list is given, only process those categories
            if preserve and cat_name not in preserve:
                continue

            keywords = cat_data.get("keywords")

            # Keyword-based matching (check if any keyword appears in text)
            if keywords:
                for kw in keywords:
                    if re.search(r"\b" + re.escape(kw.lower()) + r"\b", lower_text):
                        memories.append(
                            Memory(
                                key=cat_name,
                                value=kw.lower(),
                                source_role=role,
                                timestamp_index=idx,
                            )
                        )

            # Pattern-based matching
            for compiled_re in self._compiled_patterns.get(cat_name, []):
                match = compiled_re.search(text)
                if match and match.group(1).strip():
                    captured = match.group(1).strip().rstrip(".,!?")
                    # Skip overly long captures (likely false positives)
                    if len(captured) > 150:
                        continue

                    # If this category has keywords, only accept exact matches
                    if keywords:
                        if captured.lower() not in [kw.lower() for kw in keywords]:
                            continue

                    # Strip trailing prepositions/articles and everything after from task values
                    if cat_name == "task":
                        captured = re.sub(
                            r"\s+(to|for|on|in|at|the|a|an)\b.*$",
                            "",
                            captured,
                            flags=re.IGNORECASE,
                        ).strip()
                        if not captured:
                            continue

                    memories.append(
                        Memory(
                            key=cat_name,
                            value=captured,
                            source_role=role,
                            timestamp_index=idx,
                        )
                    )

        return memories

    # ------------------------------------------------------------------
    # Layer 2 — spaCy NER
    # ------------------------------------------------------------------

    def _extract_ner(
        self, text: str, idx: int, role: str
    ) -> List[Memory]:
        """Extract memories from named entities recognized by spaCy."""
        if self.nlp is None:
            return []

        memories: List[Memory] = []
        doc = self.nlp(text)

        for ent in doc.ents:
            mapped_key = self._entity_map.get(ent.label_)
            if mapped_key:
                memories.append(
                    Memory(
                        key=mapped_key,
                        value=ent.text,
                        source_role=role,
                        timestamp_index=idx,
                    )
                )

        return memories

    # ------------------------------------------------------------------
    # Layer 3 — Dependency Parsing
    # ------------------------------------------------------------------

    def _extract_dependencies(
        self, text: str, idx: int, role: str
    ) -> List[Memory]:
        """Extract action intents from dependency parsing."""
        if self.nlp is None:
            return []

        memories: List[Memory] = []
        doc = self.nlp(text)

        for token in doc:
            if token.lemma_.lower() not in self._dep_verbs:
                continue

            verb = token.lemma_.lower()
            objects: List[str] = []

            for child in token.children:
                if child.dep_ == "dobj":
                    # Get the full noun chunk for richer context
                    obj_text = child.text
                    objects.append(obj_text)

                    # Look for prepositional objects connected to the dobj
                    for grandchild in child.children:
                        if grandchild.dep_ == "prep":
                            for pobj in grandchild.children:
                                if pobj.dep_ == "pobj":
                                    objects.append(pobj.text)

            if objects:
                value = f"{verb} {' '.join(objects)}"
                memories.append(
                    Memory(
                        key="action_intent",
                        value=value,
                        source_role=role,
                        timestamp_index=idx,
                    )
                )

        return memories

    # ------------------------------------------------------------------
    # Scoring & Deduplication
    # ------------------------------------------------------------------

    def _score_memories(
        self,
        memories: List[Memory],
        messages: List[Message],
        total: int,
    ) -> None:
        """Compute importance score for each extracted memory in-place."""
        for mem in memories:
            recency = (mem.timestamp_index + 1) / max(total, 1)

            # Emphasis score — check the source message for emphasis signals
            emphasis = 0.0
            if mem.timestamp_index < len(messages):
                source_text = messages[mem.timestamp_index].content.lower()
                for phrase in self._emphasis.get("high", []):
                    if phrase in source_text:
                        emphasis = 0.3
                        break
                if emphasis == 0.0:
                    for phrase in self._emphasis.get("medium", []):
                        if phrase in source_text:
                            emphasis = 0.15
                            break

            # Role score
            role_score = {"user": 1.0, "assistant": 0.7}.get(
                mem.source_role, 0.4
            )

            # Frequency component starts at 0.5
            freq_component = 0.5

            importance = (
                0.35 * recency
                + 0.30 * freq_component
                + 0.20 * emphasis
                + 0.15 * role_score
            )

            mem.importance = max(round(importance, 3), 0.50)

    def _deduplicate(self, memories: List[Memory]) -> List[Memory]:
        """Remove duplicate memories (fuzzy matching), keeping highest importance and longest string."""
        seen: List[Memory] = []

        for mem in memories:
            norm_val = _normalize_value(mem.value)
            found_duplicate = False

            for i, existing in enumerate(seen):
                if mem.key != existing.key:
                    continue

                exist_norm = _normalize_value(existing.value)
                
                # Check for substring match or high similarity
                ratio = difflib.SequenceMatcher(None, norm_val, exist_norm).ratio()
                is_substring = (norm_val in exist_norm) or (exist_norm in norm_val)
                
                if ratio > 0.65 or is_substring:
                    found_duplicate = True
                    # Keep the highest importance
                    seen[i].importance = max(seen[i].importance, mem.importance)
                    # Keep the longest, most descriptive value
                    if len(mem.value) > len(existing.value):
                        seen[i].value = mem.value
                    break

            if not found_duplicate:
                seen.append(mem)

        return seen
