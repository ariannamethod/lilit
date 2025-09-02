import random
import pathlib
from typing import List, Iterable
from memory import get_recent_messages, tokenize

DATA_PATH = pathlib.Path(__file__).with_name('smalltalk.md')


def dataset_words() -> List[str]:
    if DATA_PATH.exists():
        text = DATA_PATH.read_text(encoding='utf-8')
        return list(set(tokenize(text)))
    return []


def retrieve(query_words: Iterable[str], distance: float = 0.5) -> List[str]:
    base = set(dataset_words())
    for msg in get_recent_messages():
        base.update(tokenize(msg))
    base = [w for w in base if w not in set(query_words)]
    random.shuffle(base)
    k = max(1, int(len(base) * distance))
    return base[:k]
