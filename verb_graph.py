import re
from collections import defaultdict, Counter
from typing import Dict

class VerbGraph:
    """Track verbs and trailing punctuation with edge weights."""

    def __init__(self) -> None:
        # verb -> punctuation -> count
        self.edges: Dict[str, Counter] = defaultdict(Counter)

    def add_sentence(self, text: str) -> None:
        """Update graph counts for verbs followed by punctuation.

        Parameters
        ----------
        text: str
            Sentence to process.
        """
        tokens = re.findall(r"\b\w+\b|[.!?]", text.lower())
        for i in range(len(tokens) - 1):
            word, nxt = tokens[i], tokens[i + 1]
            if nxt in ".!?":
                self.edges[word][nxt] += 1

    def preferred_punct(self, verb: str) -> str:
        """Return punctuation most frequently following *verb*."""
        counts = self.edges.get(verb.lower())
        if counts:
            return counts.most_common(1)[0][0]
        return "."
