import os
import sys

# Ensure project root is on the import path for direct module imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from memory import tokenize
from smalltalk.grammar import tag_tokens

def test_infinitive_detection():
    tokens = tokenize("I love to eat")
    tagged = tag_tokens(tokens)
    assert tagged == [
        ("i", "PRON"),
        ("love", "VERB"),
        ("to", "TO"),
        ("eat", "INF"),
    ]


def test_gerund_detection():
    tokens = tokenize("I love playing")
    tagged = tag_tokens(tokens)
    assert tagged == [
        ("i", "PRON"),
        ("love", "VERB"),
        ("playing", "GERUND"),
    ]
