"""Message compressor using extractive summarization (sumy LSA) with fallback."""

import warnings
from typing import Dict, List, Optional

from ctxintel.models import Message

try:
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.lsa import LsaSummarizer
    from sumy.nlp.stemmers import Stemmer
    _HAS_SUMY = True
except ImportError:
    _HAS_SUMY = False
    warnings.warn(
        "sumy not found. Compression will use a simpler sentence-based fallback. "
        "Install with: pip install sumy",
        stacklevel=2,
    )


class Compressor:
    """Compresses low-importance messages into a summary while preserving critical ones.

    Uses sumy's LSA summarizer for extractive summarization (zero AI API calls).
    Falls back to a simple first-N-sentences approach if sumy is unavailable.
    """

    def compress(
        self,
        messages: List[Message],
        preserve: Optional[List[str]] = None,
        threshold: float = 0.4,
    ) -> List[Message]:
        """Compress messages below the importance threshold into a summary.

        Args:
            messages: Ranked list of Message objects.
            preserve: Category names to preserve (unused — preserved flag is on Message).
            threshold: Messages below this importance score get compressed.

        Returns:
            List of messages with low-importance ones replaced by a summary.
        """
        keep_msgs = [m for m in messages if m.importance >= threshold or m.preserved]
        compress_msgs = [m for m in messages if m.importance < threshold and not m.preserved]

        if not compress_msgs:
            return keep_msgs

        combined_text = " ".join(m.content for m in compress_msgs)

        # If the compressed content is very short, just drop it —
        # a summary message would be larger than the original filler
        if len(combined_text) < 50:
            return keep_msgs

        summary = self._summarize(combined_text, len(compress_msgs))

        # Only inject summary if it's actually shorter than the combined original
        if len(summary) >= len(combined_text):
            return keep_msgs

        summary_msg = Message(
            role="system",
            content=f"[Earlier context summary]: {summary}",
            importance=0.85,
            preserved=True,
        )
        return [summary_msg] + keep_msgs

    def estimate_reduction(self, messages: List[Message], threshold: float = 0.4) -> Dict:
        """Preview compression stats without actually compressing.

        Args:
            messages: Ranked list of Message objects.
            threshold: Importance threshold for compression.

        Returns:
            Dict with total_messages, will_keep, will_compress, estimated_reduction_pct.
        """
        keep = sum(1 for m in messages if m.importance >= threshold or m.preserved)
        compress = sum(1 for m in messages if m.importance < threshold and not m.preserved)
        total = len(messages)
        return {
            "total_messages": total,
            "will_keep": keep,
            "will_compress": compress,
            "estimated_reduction_pct": round(compress / max(total, 1) * 100, 1),
        }

    def _summarize(self, text: str, msg_count: int) -> str:
        """Summarize text using sumy LSA or a sentence-based fallback."""
        if _HAS_SUMY:
            return self._summarize_lsa(text, msg_count)
        return self._summarize_fallback(text)

    def _summarize_lsa(self, text: str, msg_count: int) -> str:
        """Use sumy's LSA summarizer for extractive summarization."""
        try:
            sentence_count = max(3, min(7, msg_count // 3))
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            stemmer = Stemmer("english")
            summarizer = LsaSummarizer(stemmer)
            sentences = summarizer(parser.document, sentence_count)
            summary = " ".join(str(s) for s in sentences)
            if not summary.strip():
                return self._summarize_fallback(text)
            return summary
        except Exception as exc:
            warnings.warn(f"sumy LSA failed ({exc}). Using fallback.", stacklevel=2)
            return self._summarize_fallback(text)

    def _summarize_fallback(self, text: str) -> str:
        """Simple fallback: take the first 5 sentences."""
        sentences = text.split(". ")
        selected = sentences[:5]
        result = ". ".join(selected)
        if not result.endswith("."):
            result += "."
        return result
