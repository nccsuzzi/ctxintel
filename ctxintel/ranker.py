"""Message importance ranker using TF-IDF, recency, signals, and semantic uniqueness."""

import warnings
from typing import List

import yaml

from ctxintel.models import Message

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False
    warnings.warn(
        "scikit-learn not found. TF-IDF scoring and semantic uniqueness "
        "will be disabled. Install with: pip install scikit-learn",
        stacklevel=2,
    )

from importlib import resources as importlib_resources


def _load_emphasis_signals() -> dict:
    """Load emphasis signal patterns from patterns.yaml."""
    try:
        data_pkg = importlib_resources.files("ctxintel") / "data" / "patterns.yaml"
        raw = data_pkg.read_text(encoding="utf-8")
        config = yaml.safe_load(raw)
        return config.get("emphasis_signals", {})
    except Exception:
        return {"high": [], "medium": []}


# Signal keywords used for pattern scoring
_DECISION_WORDS = {"decided", "decision", "chose", "chosen", "settled", "confirmed"}
_PREFERENCE_WORDS = {"prefer", "favorite", "favourite", "like", "love", "enjoy"}


class Ranker:
    """Scores every message using a weighted combination of four signals.

    score = 0.30 * tfidf + 0.30 * recency + 0.20 * signal_pattern + 0.20 * uniqueness

    System messages always receive importance 1.0.
    Short messages (< 15 chars stripped) receive a -0.15 penalty.
    All scores are clamped to [0.0, 1.0] and rounded to 3 decimal places.
    """

    def __init__(self) -> None:
        self._emphasis_signals = _load_emphasis_signals()

    def rank(self, messages: List[Message]) -> List[Message]:
        """Score every message in the list and return them with updated importance.

        Args:
            messages: List of Message objects to score.

        Returns:
            The same list of Message objects with their importance fields updated.
        """
        if not messages:
            return messages

        total = len(messages)
        contents = [m.content for m in messages]

        tfidf_scores = self._compute_tfidf_scores(contents)
        uniqueness_scores = self._compute_uniqueness_scores(contents)

        for i, msg in enumerate(messages):
            # System messages always get max importance
            if msg.role == "system":
                msg.importance = 1.0
                continue

            recency = i / max(total - 1, 1)
            signal = self._compute_signal_score(msg.content)
            tfidf = tfidf_scores[i] if tfidf_scores else 0.0
            unique = uniqueness_scores[i] if uniqueness_scores else 0.5

            score = (
                0.30 * tfidf
                + 0.30 * recency
                + 0.20 * signal
                + 0.20 * unique
            )

            # Short-message penalty
            if len(msg.content.strip()) < 15:
                score -= 0.15

            # Clamp and round
            score = max(0.0, min(1.0, score))
            msg.importance = round(score, 3)

        return messages

    # ------------------------------------------------------------------
    # Private scoring helpers
    # ------------------------------------------------------------------

    def _compute_tfidf_scores(self, contents: List[str]) -> List[float]:
        """Compute TF-IDF row sums normalized to [0, 1]."""
        if not _HAS_SKLEARN or len(contents) < 2:
            return [0.0] * len(contents)

        try:
            vectorizer = TfidfVectorizer(
                stop_words="english", max_features=500
            )
            matrix = vectorizer.fit_transform(contents)
            row_sums = matrix.sum(axis=1).A1  # flatten to 1-D array
            max_val = row_sums.max() if row_sums.max() > 0 else 1.0
            return (row_sums / max_val).tolist()
        except Exception:
            return [0.0] * len(contents)

    def _compute_uniqueness_scores(self, contents: List[str]) -> List[float]:
        """Compute semantic uniqueness: 1.0 - max cosine similarity to earlier messages."""
        if not _HAS_SKLEARN or len(contents) < 2:
            return [1.0] * len(contents)

        try:
            vectorizer = TfidfVectorizer(
                stop_words="english", max_features=500
            )
            matrix = vectorizer.fit_transform(contents)
            scores: List[float] = [1.0]  # first message is always unique

            for i in range(1, len(contents)):
                sim = cosine_similarity(matrix[i], matrix[:i])
                max_sim = float(sim.max())
                scores.append(round(1.0 - max_sim, 4))

            return scores
        except Exception:
            return [1.0] * len(contents)

    def _compute_signal_score(self, text: str) -> float:
        """Score a message by emphasis signals, questions, decisions, and preferences."""
        score = 0.0
        lower = text.lower()

        # Emphasis signals from YAML
        for phrase in self._emphasis_signals.get("high", []):
            if phrase in lower:
                score += 0.15

        for phrase in self._emphasis_signals.get("medium", []):
            if phrase in lower:
                score += 0.08

        # Question marks indicate information-seeking
        if "?" in text:
            score += 0.05

        # Decision words
        words = set(lower.split())
        if words & _DECISION_WORDS:
            score += 0.10

        # Preference words
        if words & _PREFERENCE_WORDS:
            score += 0.08

        return min(score, 0.40)
