import sqlite3
import math
import time
import re
from typing import List, Tuple, Dict

DB_PATH = 'memory.db'

def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute('CREATE TABLE IF NOT EXISTS dialog (id INTEGER PRIMARY KEY AUTOINCREMENT, message TEXT, ts REAL)')
        c.execute('CREATE TABLE IF NOT EXISTS vocab (word TEXT PRIMARY KEY, count INTEGER)')
        conn.commit()

def tokenize(text: str) -> List[str]:
    return [w.lower() for w in re.findall(r"[a-zA-Z']+", text)]

def log_message(message: str) -> None:
    words = tokenize(message)
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute('INSERT INTO dialog (message, ts) VALUES (?, ?)', (message, time.time()))
        for w in words:
            c.execute('INSERT INTO vocab (word, count) VALUES (?, 1) ON CONFLICT(word) DO UPDATE SET count=count+1', (w,))
        conn.commit()

def get_recent_messages(n: int = 20) -> List[str]:
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute('SELECT message FROM dialog ORDER BY id DESC LIMIT ?', (n,))
        return [row[0] for row in c.fetchall()]

def get_vocab() -> Dict[str, int]:
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute('SELECT word, count FROM vocab')
        return {w: cnt for w, cnt in c.fetchall()}

def metrics(words: List[str], vocab: Dict[str, int]) -> Tuple[float, float, str]:
    total = sum(vocab.values()) or 1
    probs = [vocab.get(w, 1) / total for w in words] or [1 / total]
    entropy = -sum(p * math.log2(p) for p in probs)
    perplexity = 2 ** entropy
    resonance_word = min(words, key=lambda w: vocab.get(w, 1)) if words else ''
    return entropy, perplexity, resonance_word
