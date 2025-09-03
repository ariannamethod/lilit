import random
import pathlib
from typing import List, Iterable, Optional
from memory import get_recent_messages, tokenize

DATA_PATH = pathlib.Path(__file__).with_name('smalltalk.md')

# Cached dataset words and the source file's modification time.
_DATASET_CACHE: List[str] = []
_DATASET_MTIME: Optional[float] = None


def dataset_words() -> List[str]:
    """Return cached word list from ``smalltalk.md``.

    The file is read only once and reloaded when its modification time
    changes. Subsequent calls return the cached value.
    """

    global _DATASET_CACHE, _DATASET_MTIME

    if DATA_PATH.exists():
        mtime = DATA_PATH.stat().st_mtime
        if _DATASET_MTIME != mtime:
            text = DATA_PATH.read_text(encoding='utf-8')
            _DATASET_CACHE = list(set(tokenize(text)))
            _DATASET_MTIME = mtime
    else:
        _DATASET_CACHE = []
        _DATASET_MTIME = None
    return _DATASET_CACHE


def retrieve(query_words: Iterable[str], distance: float = 0.5) -> List[str]:
    base = set(dataset_words())
    for msg in get_recent_messages():
        base.update(tokenize(msg))
    base = [w for w in base if w not in set(query_words)]
    random.shuffle(base)
    k = max(1, int(len(base) * distance))
    return base[:k]
