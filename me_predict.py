"""
me_predict.py - Self-prediction module for the ME engine.

This module enables the model to predict its own next user message based on:
1. Previous user message
2. Its own reply 
3. Resonance metrics

Uses SQLite as the primary engine for scoring and selection.
"""

import sqlite3
import time
from typing import List, Optional
from memory import DB_PATH, tokenize, get_vocab, metrics, get_recent_messages


# Scoring weights for candidate tokens
W_BIGRAM = 1.0      # Weight for bigram transitions from last token
W_RES = 0.7         # Weight for resonance-based bigram transitions  
W_RARE = 0.35       # Weight for token rarity (1/count)
W_CTX = 0.15        # Weight for context overlap


def ensure_schema() -> None:
    """Create the prediction table if it doesn't exist."""
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS prediction (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                input TEXT NOT NULL,
                reply TEXT NOT NULL,
                predicted TEXT NOT NULL,
                score REAL NOT NULL,
                resonance TEXT
            )
        ''')
        conn.commit()


def _get_target_length(input_message: str, self_reply: str) -> int:
    """Derive target length from entropy/perplexity similar to me.Engine._lengths."""
    words = tokenize(input_message + " " + self_reply)
    vocab = get_vocab()
    entropy, perplexity, _ = metrics(words, vocab)
    
    # Similar to me.Engine._lengths but simplified for single length
    base = 6 + int(entropy) % 5  # 6-10 typical range
    length = max(4, min(14, base))  # Ensure 4-14 range
    return length


def _build_context_cte(input_message: str, self_reply: str) -> str:
    """Build SQL CTE for context words from input + reply + recent messages."""
    # Get context from input and reply
    context_words = set(tokenize(input_message) + tokenize(self_reply))
    
    # Optionally add recent messages (last 10)
    recent_msgs = get_recent_messages(10)
    for msg in recent_msgs:
        context_words.update(tokenize(msg))
    
    # Build VALUES clause for context CTE
    if not context_words:
        return "ctx(word) AS (SELECT '' WHERE 0)"  # Empty CTE
    
    values = ", ".join(f"('{word.replace(chr(39), chr(39)+chr(39))}')" for word in context_words)
    return f"ctx(word) AS (VALUES {values})"


def _score_candidates_query(prev_token: str, resonance_word: str, used_tokens: List[str]) -> str:
    """Build SQL query to score candidate tokens."""
    used_filter = ""
    if used_tokens:
        escaped_used = [f"'{tok.replace(chr(39), chr(39)+chr(39))}'" for tok in used_tokens]
        used_filter = f"AND candidates.word NOT IN ({', '.join(escaped_used)})"
    
    prev_escaped = prev_token.replace(chr(39), chr(39)+chr(39))
    res_escaped = resonance_word.replace(chr(39), chr(39)+chr(39))
    
    return f'''
        WITH {_build_context_cte("", "")},
        candidates AS (
            -- Bigram candidates from previous token
            SELECT b.w2 as word, {W_BIGRAM} * b.count as bigram_score, 0 as res_score
            FROM bigram b 
            WHERE b.w1 = '{prev_escaped}'
            
            UNION ALL
            
            -- Bigram candidates from resonance word  
            SELECT b.w2 as word, 0 as bigram_score, {W_RES} * b.count as res_score
            FROM bigram b
            WHERE b.w1 = '{res_escaped}'
            AND '{resonance_word}' != ''
            
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
    '''


def predict_tokens(input_message: str, self_reply: str, n: int = 12) -> List[str]:
    """
    Returns the list of predicted tokens only (for introspection and testing).
    
    Args:
        input_message: The previous user message
        self_reply: The bot's reply to that message
        n: Number of tokens to predict
        
    Returns:
        List of predicted token strings
    """
    ensure_schema()
    
    # Get context and resonance info
    reply_words = tokenize(self_reply) 
    input_words = tokenize(input_message)
    all_words = input_words + reply_words
    vocab = get_vocab()
    
    if not all_words:
        return ['hmm'] * min(n, 4)  # Fallback for empty input
        
    _, _, resonance_word = metrics(all_words, vocab)
    
    # Start from last token of self_reply, or resonance word if no reply
    prev_token = reply_words[-1] if reply_words else resonance_word
    if not prev_token:
        prev_token = resonance_word or 'hmm'
    
    tokens = []
    used_tokens = []
    
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        
        for i in range(n):
            # Try to get next token using SQL scoring
            query = _score_candidates_query(prev_token, resonance_word, used_tokens)
            try:
                # Replace context CTE with actual context
                context_words = set(tokenize(input_message) + tokenize(self_reply))
                recent_msgs = get_recent_messages(10)
                for msg in recent_msgs:
                    context_words.update(tokenize(msg))
                
                if context_words:
                    values = ", ".join(f"('{word.replace(chr(39), chr(39)+chr(39))}')" for word in context_words)
                    context_cte = f"ctx(word) AS (VALUES {values})"
                else:
                    context_cte = "ctx(word) AS (SELECT '' WHERE 0)"
                
                query = query.replace(_build_context_cte("", ""), context_cte)
                
                c.execute(query)
                result = c.fetchone()
                
                if result and result[0]:
                    token = result[0]
                    tokens.append(token)
                    used_tokens.append(token)
                    prev_token = token
                else:
                    # Fallback: pick a rare word from vocab
                    c.execute("SELECT word FROM vocab ORDER BY count ASC LIMIT 1")
                    fallback = c.fetchone()
                    token = fallback[0] if fallback else 'hmm'
                    tokens.append(token)
                    used_tokens.append(token)
                    prev_token = token
                    
            except sqlite3.Error:
                # Final fallback
                token = 'hmm'
                tokens.append(token)
                used_tokens.append(token)
                prev_token = token
    
    return tokens


def predict_next(input_message: str, self_reply: str, *, max_len: Optional[int] = None) -> str:
    """
    Returns a single predicted next message string.
    
    Args:
        input_message: The previous user message
        self_reply: The bot's reply to that message  
        max_len: Maximum length in tokens (if None, derived from metrics)
        
    Returns:
        Predicted next user message as a string
    """
    ensure_schema()
    
    # Determine target length
    if max_len is None:
        target_length = _get_target_length(input_message, self_reply)
    else:
        target_length = max(4, max_len)  # Ensure at least 4 tokens
    
    # Get predicted tokens
    tokens = predict_tokens(input_message, self_reply, target_length)
    
    if not tokens:
        tokens = ['hmm']
    
    # Post-process tokens
    # Replace single-character fragments at the end
    if len(tokens) > 0 and len(tokens[-1]) == 1:
        tokens[-1] = 'hmm'
    
    # Capitalize first token
    if tokens:
        tokens[0] = tokens[0].capitalize()
    
    # Join and add period
    predicted_text = ' '.join(tokens)
    if not predicted_text.endswith('.'):
        predicted_text += '.'
    
    # Calculate mean score for audit (simple approximation)
    # For now, use a placeholder score based on token count and context
    vocab = get_vocab()
    all_context_words = tokenize(input_message + " " + self_reply)
    _, _, resonance_word = metrics(all_context_words, vocab)
    
    # Simple score: longer predictions score higher, rare words score higher
    base_score = len(tokens) * 0.1
    for token in tokens:
        token_count = vocab.get(token, 1)
        base_score += 1.0 / (token_count + 1)
    
    mean_score = base_score / len(tokens) if tokens else 0.0
    
    # Store prediction in database
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute('''
            INSERT INTO prediction (ts, input, reply, predicted, score, resonance)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (time.time(), input_message, self_reply, predicted_text, mean_score, resonance_word))
        conn.commit()
    
    return predicted_text


if __name__ == '__main__':
    """CLI for manual testing."""
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
