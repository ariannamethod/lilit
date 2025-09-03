"""
me2me.py - Ephemeral Micro-Transformer

Creates per-dialog, CPU-only "ephemeral micro-transformers" that build from conversation context,
generate text using attention mechanisms, and dissolve back into audit logs.

Each session instantiates a temporary transformer with deterministic weights derived from context.
The transformer uses pure Python multi-head attention with small dimensions for CPU efficiency.
Sessions persist minimal specs in SQLite and regenerate weights on-demand from seeds.
"""

import asyncio
import sqlite3
import hashlib
import random
import time
import math
from typing import List, Dict, Optional, Tuple, Any
from memory import DB_PATH, tokenize, get_vocab, get_recent_messages


def ensure_schema() -> None:
    """Create me2me_session and me2me_log tables if they don't exist."""
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        
        # Session table: stores ephemeral transformer specs
        c.execute("""
            CREATE TABLE IF NOT EXISTS me2me_session (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                seed TEXT NOT NULL,
                ttl REAL NOT NULL,
                heads INTEGER NOT NULL,
                d_model INTEGER NOT NULL,
                context_hash TEXT NOT NULL,
                merged INTEGER NOT NULL DEFAULT 0
            )
        """)
        
        # Log table: stores events and activities for sessions
        c.execute("""
            CREATE TABLE IF NOT EXISTS me2me_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                ts REAL NOT NULL,
                event TEXT NOT NULL,
                data TEXT,
                FOREIGN KEY(session_id) REFERENCES me2me_session(id)
            )
        """)
        
        conn.commit()


def _compute_context_signature(context: List[str]) -> str:
    """Compute deterministic seed from context messages."""
    # Tokenize all context messages and create signature
    all_tokens = []
    for msg in context:
        all_tokens.extend(tokenize(msg))
    
    # Create deterministic hash from tokens
    token_string = "|".join(sorted(all_tokens))
    return hashlib.sha1(token_string.encode()).hexdigest()[:16]


def _find_existing_session(context_hash: str, now: float) -> Optional[int]:
    """Return existing session id if context_hash matches and ttl is valid."""
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute(
            "SELECT id FROM me2me_session WHERE context_hash = ? AND ttl > ?",
            (context_hash, now),
        )
        row = c.fetchone()
        return row[0] if row else None


def start_session(context: Optional[List[str]] = None, ttl_seconds: int = 600,
                 heads: int = 2, d_model: int = 16) -> int:
    """
    Start a new ephemeral micro-transformer session.
    
    Args:
        context: List of messages for context, or None to use recent dialog
        ttl_seconds: Time-to-live for the session
        heads: Number of attention heads
        d_model: Model dimension
        
    Returns:
        Session ID
    """
    ensure_schema()
    
    # Build context from provided messages or recent dialog
    if context is None:
        context = get_recent_messages(8)
    
    if not context:
        context = ["hello", "welcome"]  # Fallback if no context available
    
    # Compute deterministic seed from context
    context_hash = _compute_context_signature(context)
    seed = context_hash

    now = time.time()
    existing = _find_existing_session(context_hash, now)
    if existing is not None:
        return existing

    ttl = now + ttl_seconds

    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO me2me_session (ts, seed, ttl, heads, d_model, context_hash, merged)
            VALUES (?, ?, ?, ?, ?, ?, 0)
        """,
            (now, seed, ttl, heads, d_model, context_hash),
        )

        session_id = c.lastrowid

        # Log session creation
        top_tokens = []
        for msg in context[:3]:  # First few messages for summary
            top_tokens.extend(tokenize(msg)[:5])  # Top tokens from each

        summary = (
            f"Started session: heads={heads}, d_model={d_model}, ttl={ttl_seconds}s, "
            f"top_tokens={top_tokens[:10]}"
        )
        c.execute(
            """
            INSERT INTO me2me_log (session_id, ts, event, data)
            VALUES (?, ?, 'session_start', ?)
        """,
            (session_id, now, summary),
        )

        conn.commit()

    return session_id


async def spawn_session(context: Optional[List[str]] = None, ttl_seconds: int = 600,
                        heads: int = 2, d_model: int = 16) -> int:
    """Start a session without blocking the event loop.

    Args mirror :func:`start_session`.

    Returns:
        Awaitable session ID.
    """
    ensure_schema()

    if context is None:
        context = await asyncio.to_thread(get_recent_messages, 8)

    if not context:
        context = ["hello", "welcome"]

    context_hash = _compute_context_signature(context)
    now = time.time()
    existing = await asyncio.to_thread(_find_existing_session, context_hash, now)
    if existing is not None:
        return existing

    return await asyncio.to_thread(start_session, context, ttl_seconds, heads, d_model)


def _generate_weights(seed: str, d_model: int, heads: int) -> Dict[str, Any]:
    """Generate deterministic transformer weights from seed."""
    rng = random.Random(seed)
    
    weights = {}
    head_dim = d_model // heads
    
    # Generate weights for each attention head
    for h in range(heads):
        # Query, Key, Value projection matrices
        weights[f'Wq_{h}'] = [[rng.gauss(0, 0.1) for _ in range(head_dim)] for _ in range(d_model)]
        weights[f'Wk_{h}'] = [[rng.gauss(0, 0.1) for _ in range(head_dim)] for _ in range(d_model)]
        weights[f'Wv_{h}'] = [[rng.gauss(0, 0.1) for _ in range(head_dim)] for _ in range(d_model)]
    
    # Output projection matrix
    weights['Wo'] = [[rng.gauss(0, 0.1) for _ in range(d_model)] for _ in range(d_model)]
    
    return weights


def _matrix_multiply(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Simple matrix multiplication."""
    if not A or not B or len(A[0]) != len(B):
        return []
    
    result = [[0.0 for _ in range(len(B[0]))] for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    return result


def _vector_matrix_multiply(vec: List[float], matrix: List[List[float]]) -> List[float]:
    """Multiply vector by matrix."""
    if not vec or not matrix or len(vec) != len(matrix):
        return []
    
    result = [0.0] * len(matrix[0])
    for i, val in enumerate(vec):
        for j in range(len(matrix[0])):
            result[j] += val * matrix[i][j]
    return result


def _softmax(values: List[float]) -> List[float]:
    """Compute softmax of values."""
    if not values:
        return []
    
    max_val = max(values)
    exp_vals = [math.exp(x - max_val) for x in values]
    sum_exp = sum(exp_vals)
    
    if sum_exp == 0:
        return [1.0 / len(values)] * len(values)
    
    return [x / sum_exp for x in exp_vals]


def _create_embeddings(tokens: List[str], d_model: int) -> List[List[float]]:
    """Create token embeddings using deterministic function of token and vocab stats."""
    vocab = get_vocab()
    embeddings = []
    
    for token in tokens:
        # Create embedding from token hash and vocab frequency
        token_hash = hashlib.md5(token.encode()).hexdigest()
        
        # Use vocab count to influence embedding
        count = vocab.get(token, 1)
        freq_factor = 1.0 / math.sqrt(count + 1)
        
        # Generate embedding vector
        embedding = []
        for i in range(d_model):
            # Mix token hash with position to create diverse embeddings
            seed_val = int(token_hash[i % len(token_hash)], 16) + i
            rng = random.Random(seed_val)
            val = rng.gauss(0, 1) * freq_factor
            embedding.append(val)
        
        embeddings.append(embedding)
    
    return embeddings


def _attention_layer(embeddings: List[List[float]], weights: Dict[str, Any], 
                    heads: int, d_model: int) -> List[List[float]]:
    """Apply multi-head attention to embeddings."""
    if not embeddings:
        return []
    
    seq_len = len(embeddings)
    head_dim = d_model // heads
    
    # Collect outputs from all heads
    head_outputs = []
    
    for h in range(heads):
        # Get weights for this head
        Wq = weights[f'Wq_{h}']
        Wk = weights[f'Wk_{h}']
        Wv = weights[f'Wv_{h}']
        
        # Compute Q, K, V for all positions
        Q = [_vector_matrix_multiply(emb, Wq) for emb in embeddings]
        K = [_vector_matrix_multiply(emb, Wk) for emb in embeddings]
        V = [_vector_matrix_multiply(emb, Wv) for emb in embeddings]
        
        # Compute attention scores and apply attention
        head_output = []
        for i in range(seq_len):
            scores = []
            for j in range(seq_len):
                # Scaled dot-product attention
                score = sum(Q[i][k] * K[j][k] for k in range(head_dim))
                score /= math.sqrt(head_dim)
                scores.append(score)
            
            # Apply softmax to get attention weights
            attn_weights = _softmax(scores)
            
            # Weighted sum of values
            output_vec = [0.0] * head_dim
            for j in range(seq_len):
                for k in range(head_dim):
                    output_vec[k] += attn_weights[j] * V[j][k]
            
            head_output.append(output_vec)
        
        head_outputs.append(head_output)
    
    # Concatenate heads and apply output projection
    concat_output = []
    for i in range(seq_len):
        concat_vec = []
        for h in range(heads):
            concat_vec.extend(head_outputs[h][i])
        
        # Apply output projection
        output_vec = _vector_matrix_multiply(concat_vec, weights['Wo'])
        
        # Simple residual connection
        for j in range(d_model):
            output_vec[j] += embeddings[i][j]
        
        # Simple layer norm proxy (scale to unit max instead of true layer norm)
        max_val = max(abs(x) for x in output_vec) or 1.0
        output_vec = [x / max_val for x in output_vec]
        
        concat_output.append(output_vec)
    
    return concat_output


def generate(session_id: int, prompt: str, max_new_tokens: int = 24) -> str:
    """
    Generate text using the ephemeral micro-transformer.
    
    Args:
        session_id: ID of the session to use
        prompt: Input prompt to generate from
        max_new_tokens: Maximum number of tokens to generate
        
    Returns:
        Generated text string
    """
    start_time = time.time()
    
    # Get session info
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("""
            SELECT seed, heads, d_model, ttl, merged 
            FROM me2me_session WHERE id = ?
        """, (session_id,))
        
        row = c.fetchone()
        if not row:
            raise ValueError(f"Session {session_id} not found")
        
        seed, heads, d_model, ttl, merged = row
        
        if merged:
            raise ValueError(f"Session {session_id} has been dissolved")
        
        if time.time() > ttl:
            raise ValueError(f"Session {session_id} has expired")
    
    # Generate weights deterministically
    weights = _generate_weights(seed, d_model, heads)
    
    # Tokenize prompt
    prompt_tokens = tokenize(prompt)
    if not prompt_tokens:
        prompt_tokens = ["hello"]
    
    generated_tokens = []
    current_tokens = prompt_tokens.copy()
    
    vocab = get_vocab()
    vocab_list = list(vocab.keys())
    
    if not vocab_list:
        vocab_list = ["hello", "world", "yes", "no", "maybe"]
    
    # Generate tokens one by one
    for _ in range(max_new_tokens):
        # Create embeddings for current sequence
        embeddings = _create_embeddings(current_tokens, d_model)
        
        # Apply attention layer
        attended = _attention_layer(embeddings, weights, heads, d_model)
        
        if not attended:
            break
        
        # Use last position output for next token prediction
        last_output = attended[-1]
        
        # Score vocabulary tokens based on similarity to output
        scores = []
        for candidate in vocab_list:
            candidate_emb = _create_embeddings([candidate], d_model)[0]
            
            # Compute similarity (dot product)
            similarity = sum(last_output[i] * candidate_emb[i] for i in range(d_model))
            
            # Bonus for bigram continuation if available
            if current_tokens:
                last_token = current_tokens[-1]
                bigram_bonus = 0.0
                
                # Check bigram table for continuation probability
                with sqlite3.connect(DB_PATH) as conn:
                    c = conn.cursor()
                    c.execute("SELECT count FROM bigram WHERE w1 = ? AND w2 = ?", 
                             (last_token, candidate))
                    bigram_row = c.fetchone()
                    if bigram_row:
                        bigram_bonus = math.log(bigram_row[0] + 1) * 0.1
                
                similarity += bigram_bonus
            
            scores.append((similarity, candidate))
        
        # Select token (greedy for simplicity, could add sampling)
        if scores:
            scores.sort(reverse=True)
            next_token = scores[0][1]
        else:
            next_token = "hmm"
        
        generated_tokens.append(next_token)
        current_tokens.append(next_token)
        
        # Stop if we hit a natural ending
        if next_token in [".", "!", "?"] or len(current_tokens) > 50:
            break
    
    # Log generation event
    latency_ms = (time.time() - start_time) * 1000
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        log_data = f"prompt_len={len(prompt_tokens)}, generated={len(generated_tokens)}, latency={latency_ms:.1f}ms"
        c.execute("""
            INSERT INTO me2me_log (session_id, ts, event, data)
            VALUES (?, ?, 'generation', ?)
        """, (session_id, time.time(), log_data))
        conn.commit()
    
    return " ".join(generated_tokens)


def active_sessions() -> List[Dict[str, Any]]:
    """Return list of active (non-merged, non-expired) sessions."""
    now = time.time()
    
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("""
            SELECT id, ts, seed, ttl, heads, d_model, context_hash
            FROM me2me_session 
            WHERE merged = 0 AND ttl > ?
            ORDER BY ts DESC
        """, (now,))
        
        sessions = []
        for row in c.fetchall():
            sessions.append({
                'id': row[0],
                'ts': row[1],
                'seed': row[2],
                'ttl': row[3],
                'heads': row[4],
                'd_model': row[5],
                'context_hash': row[6],
                'expires_in': row[3] - now
            })
        
        return sessions


def ephemeral_cleanup(now: Optional[float] = None) -> int:
    """Remove or mark expired sessions. Returns number of sessions cleaned up."""
    if now is None:
        now = time.time()
    
    ensure_schema()
    
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        
        # Find expired sessions
        c.execute("SELECT id FROM me2me_session WHERE ttl < ? AND merged = 0", (now,))
        expired_ids = [row[0] for row in c.fetchall()]
        
        count = 0
        for session_id in expired_ids:
            # Log cleanup event
            c.execute("""
                INSERT INTO me2me_log (session_id, ts, event, data)
                VALUES (?, ?, 'cleanup', 'Session expired and cleaned up')
            """, (session_id, now))
            
            # Mark as merged (soft delete)
            c.execute("UPDATE me2me_session SET merged = 1 WHERE id = ?", (session_id,))
            count += 1
        
        conn.commit()
        return count


def merge_and_dissolve(session_id: int) -> None:
    """Merge session summary into logs and dissolve the session."""
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        
        # Get session info
        c.execute("""
            SELECT ts, heads, d_model, merged FROM me2me_session WHERE id = ?
        """, (session_id,))
        
        row = c.fetchone()
        if not row:
            raise ValueError(f"Session {session_id} not found")
        
        session_ts, heads, d_model, merged = row
        
        if merged:
            raise ValueError(f"Session {session_id} already dissolved")
        
        # Count generations and gather stats
        c.execute("""
            SELECT COUNT(*) FROM me2me_log 
            WHERE session_id = ? AND event = 'generation'
        """, (session_id,))
        generation_count = c.fetchone()[0]
        
        # Calculate elapsed time
        elapsed = time.time() - session_ts
        
        # Create summary
        summary = f"Session dissolved: {generation_count} generations, {elapsed:.1f}s elapsed, heads={heads}, d_model={d_model}"
        
        # Log final summary
        c.execute("""
            INSERT INTO me2me_log (session_id, ts, event, data)
            VALUES (?, ?, 'dissolve', ?)
        """, (session_id, time.time(), summary))
        
        # Mark as merged/dissolved
        c.execute("UPDATE me2me_session SET merged = 1 WHERE id = ?", (session_id,))
        
        conn.commit()


if __name__ == '__main__':
    """CLI interface for interactive usage."""
    print("ME2ME: Ephemeral Micro-Transformer")
    print("==================================")
    
    # Ensure schema exists
    ensure_schema()
    
    # Cleanup any expired sessions
    cleaned = ephemeral_cleanup()
    if cleaned > 0:
        print(f"Cleaned up {cleaned} expired sessions.")
    
    # Start a new session
    print("\nStarting new session from recent dialog...")
    try:
        session_id = start_session()
        print(f"Session {session_id} started.")
        
        while True:
            print(f"\nSession {session_id} active. Options:")
            print("  (G)enerate - Generate text from prompt")
            print("  (M)erge and dissolve - End session")
            print("  (Q)uit - Exit without dissolving")
            
            choice = input("> ").lower().strip()
            
            if choice in ['g', 'generate']:
                prompt = input("Enter prompt: ").strip()
                if prompt:
                    try:
                        result = generate(session_id, prompt)
                        print(f"Generated: {result}")
                    except Exception as e:
                        print(f"Generation error: {e}")
                else:
                    print("Empty prompt, skipping.")
            
            elif choice in ['m', 'merge']:
                try:
                    merge_and_dissolve(session_id)
                    print(f"Session {session_id} dissolved.")
                    break
                except Exception as e:
                    print(f"Dissolve error: {e}")
            
            elif choice in ['q', 'quit']:
                print(f"Exiting. Session {session_id} remains active.")
                break
            
            else:
                print("Invalid choice. Use G, M, or Q.")
    
    except Exception as e:
        print(f"Error: {e}")