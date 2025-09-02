import sqlite3
from memory import DB_PATH, tokenize, log_message


def train(user_message: str) -> None:
    log_message(user_message)
    words = tokenize(user_message)
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute('CREATE TABLE IF NOT EXISTS bigram (w1 TEXT, w2 TEXT, count INTEGER, PRIMARY KEY (w1, w2))')
        for a, b in zip(words, words[1:]):
            c.execute('INSERT INTO bigram (w1, w2, count) VALUES (?, ?, 1) ON CONFLICT(w1, w2) DO UPDATE SET count=count+1', (a, b))
        conn.commit()
