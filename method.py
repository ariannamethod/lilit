import asyncio
from typing import Optional

import aiosqlite

from memory import DB_PATH, tokenize, log_message


DB_CONN: Optional[aiosqlite.Connection] = None


async def get_conn() -> aiosqlite.Connection:
    """Return a shared aiosqlite connection and ensure schema."""
    global DB_CONN
    if DB_CONN is None:
        DB_CONN = await aiosqlite.connect(DB_PATH)
        await DB_CONN.execute(
            'CREATE TABLE IF NOT EXISTS bigram (w1 TEXT, w2 TEXT, count INTEGER, PRIMARY KEY (w1, w2))'
        )
        await DB_CONN.execute(
            'CREATE INDEX IF NOT EXISTS idx_bigram_w1_w2 ON bigram(w1, w2)'
        )
        await DB_CONN.commit()
    return DB_CONN


async def train(user_message: str) -> None:
    """Update bigram counts based on the user's message."""
    await asyncio.to_thread(log_message, user_message)
    words = tokenize(user_message)
    conn = await get_conn()
    for a, b in zip(words, words[1:]):
        await conn.execute(
            'INSERT INTO bigram (w1, w2, count) VALUES (?, ?, 1) '
            'ON CONFLICT(w1, w2) DO UPDATE SET count=count+1',
            (a, b),
        )
    await conn.commit()

