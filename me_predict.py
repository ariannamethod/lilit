"""
me_predict.py - Self-prediction module for the ME engine.

This module enables the model to predict its own next user message based on:
1. Previous user message
2. Its own reply
3. Resonance metrics

Uses SQLite as the primary engine for scoring and selection.
"""

import asyncio
import time
from typing import List, Optional, Tuple

import aiosqlite

from memory import (
    DB_PATH,
    tokenize,
    get_vocab,
    metrics,
    get_recent_messages,
)


# Scoring weights for candidate tokens
W_BIGRAM = 1.0  # Weight for bigram transitions from last token
W_RES = 0.7  # Weight for resonance-based bigram transitions
W_RARE = 0.35  # Weight for token rarity (1/count)
W_CTX = 0.15  # Weight for context overlap


async def ensure_schema(conn: aiosqlite.Connection) -> None:
    """Create the prediction table if it doesn't exist."""
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            input TEXT NOT NULL,
            reply TEXT NOT NULL,
            predicted TEXT NOT NULL,
            score REAL NOT NULL,
            resonance TEXT
        )
        """
    )
    await conn.commit()


def _build_context_words(input_message: str, self_reply: str) -> List[str]:
    """Collect context words from input, reply and recent messages."""
    context_words = set(tokenize(input_message) + tokenize(self_reply))
    recent_msgs = get_recent_messages(10)
    for msg in recent_msgs:
        context_words.update(tokenize(msg))
    return list(context_words)


def _score_candidates_query(
    prev_token: str,
    resonance_word: str,
    used_tokens: List[str],
    context_words: List[str],
) -> Tuple[str, List[str]]:
    """Build SQL query and parameters to score candidate tokens."""

    # Build context CTE
    params: List[str] = []
    if context_words:
        placeholders = ", ".join(["(?)"] * len(context_words))
        context_cte = f"ctx(word) AS (VALUES {placeholders})"
        params.extend(context_words)
    else:
        context_cte = "ctx(word) AS (SELECT '' WHERE 0)"

    # Used tokens filter
    if used_tokens:
        used_placeholders = ", ".join(["?"] * len(used_tokens))
        used_filter = f"AND candidates.word NOT IN ({used_placeholders})"
    else:
        used_filter = ""

    query = f"""
        WITH {context_cte},
        candidates AS (
            -- Bigram candidates from previous token
            SELECT b.w2 as word, {W_BIGRAM} * b.count as bigram_score, 0 as res_score
            FROM bigram b
            WHERE b.w1 = ?

            UNION ALL

            -- Bigram candidates from resonance word
            SELECT b.w2 as word, 0 as bigram_score, {W_RES} * b.count as res_score
            FROM bigram b
            WHERE b.w1 = ?
            AND ? != ''

            UNION ALL

            -- Vocab fallback candidates (rare words not in context)
            SELECT v.word, 0 as bigram_score, 0 as res_score
            FROM vocab v
            LEFT JOIN ctx c ON v.word = c.word
            WHERE c.word IS NULL  -- Not in context
            AND v.count <= 5      -- Rare words

            UNION ALL

            -- Context words as final fallback
            SELECT c.word, 0 as bigram_score, 0 as res_score
            FROM ctx c
        )
        SELECT
            candidates.word,
            candidates.bigram_score + candidates.res_score +
            {W_RARE} * (1.0 / (COALESCE(v.count, 0) + 1)) +
            {W_CTX} * CASE WHEN c.word IS NOT NULL THEN 1 ELSE 0 END as score
        FROM candidates
        LEFT JOIN vocab v ON candidates.word = v.word
        LEFT JOIN ctx c ON candidates.word = c.word
        WHERE candidates.word != ''
        {used_filter}
        ORDER BY score DESC
        LIMIT 1
    """

    params.extend([prev_token, resonance_word, resonance_word])
    if used_tokens:
        params.extend(used_tokens)

    return query, params


async def predict_tokens_async(
    input_message: str,
    self_reply: str,
    n: int = 12,
    *,
    conn: Optional[aiosqlite.Connection] = None,
) -> List[str]:
    """
    Returns the list of predicted tokens only (for introspection and testing).
    """

    own_conn = False
    if conn is None:
        conn = await aiosqlite.connect(DB_PATH)
        own_conn = True
    try:
        await ensure_schema(conn)

        # Get context and resonance info without blocking
        reply_words = tokenize(self_reply)
        input_words = tokenize(input_message)
        all_words = input_words + reply_words
        vocab = await asyncio.to_thread(get_vocab)

        if not all_words:
            return ["hmm"] * min(n, 4)

        _, _, resonance_word = metrics(all_words, vocab)

        prev_token = reply_words[-1] if reply_words else resonance_word or "hmm"

        tokens: List[str] = []
        used_tokens: List[str] = []

        c = await conn.cursor()
        for _ in range(n):
            context_words = await asyncio.to_thread(
                _build_context_words, input_message, self_reply
            )

            query, params = _score_candidates_query(
                prev_token, resonance_word, used_tokens, context_words
            )

            try:
                await c.execute(query, params)
                result = await c.fetchone()
                if result and result[0]:
                    token = result[0]
                else:
                    await c.execute(
                        "SELECT word FROM vocab ORDER BY count ASC LIMIT 1"
                    )
                    fallback = await c.fetchone()
                    token = fallback[0] if fallback else "hmm"

            except aiosqlite.Error:
                token = "hmm"

            tokens.append(token)
            used_tokens.append(token)
            prev_token = token

        return tokens
    finally:
        if own_conn:
            await conn.close()


async def predict_next_async(
    input_message: str,
    self_reply: str,
    *,
    max_len: Optional[int] = None,
) -> str:
    """Returns a single predicted next message string."""

    async with aiosqlite.connect(DB_PATH) as conn:
        await ensure_schema(conn)

        if max_len is None:
            target_length = await asyncio.to_thread(
                _get_target_length, input_message, self_reply
            )
        else:
            target_length = max(4, max_len)

        tokens = await predict_tokens_async(
            input_message, self_reply, target_length, conn=conn
        )

        if not tokens:
            tokens = ["hmm"]

        if len(tokens) > 0 and len(tokens[-1]) == 1:
            tokens[-1] = "hmm"

        if tokens:
            tokens[0] = tokens[0].capitalize()

        predicted_text = " ".join(tokens)
        if not predicted_text.endswith("."):
            predicted_text += "."

        vocab = await asyncio.to_thread(get_vocab)
        all_context_words = tokenize(input_message + " " + self_reply)
        _, _, resonance_word = metrics(all_context_words, vocab)

        base_score = len(tokens) * 0.1
        for token in tokens:
            token_count = vocab.get(token, 1)
            base_score += 1.0 / (token_count + 1)

        mean_score = base_score / len(tokens) if tokens else 0.0

        await conn.execute(
            """
            INSERT INTO prediction (ts, input, reply, predicted, score, resonance)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                time.time(),
                input_message,
                self_reply,
                predicted_text,
                mean_score,
                resonance_word,
            ),
        )
        await conn.commit()

        return predicted_text


def _get_target_length(input_message: str, self_reply: str) -> int:
    """Derive target length from entropy/perplexity similar to me.Engine._lengths."""
    words = tokenize(input_message + " " + self_reply)
    vocab = get_vocab()
    entropy, perplexity, _ = metrics(words, vocab)

    base = 6 + int(entropy) % 5  # 6-10 typical range
    length = max(4, min(14, base))  # Ensure 4-14 range
    return length


def predict_tokens(input_message: str, self_reply: str, n: int = 12) -> List[str]:
    """Synchronous wrapper for predict_tokens_async."""
    return asyncio.run(predict_tokens_async(input_message, self_reply, n))


def predict_next(
    input_message: str, self_reply: str, *, max_len: Optional[int] = None
) -> str:
    """Synchronous wrapper for predict_next_async."""
    return asyncio.run(predict_next_async(input_message, self_reply, max_len=max_len))


if __name__ == "__main__":
    print("ME Predict - Self-prediction module")
    print("Enter the last user message and your last reply:")

    try:
        input_msg = input("Last user message: ").strip()
        reply_msg = input("Your last reply: ").strip()

        if not input_msg or not reply_msg:
            print("Both messages are required.")
        else:
            prediction = predict_next(input_msg, reply_msg)
            print(f"\nPredicted next message: {prediction}")

    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"Error: {e}")

