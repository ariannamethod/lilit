from typing import List, Tuple

PRONOUNS = {"i", "you", "he", "she", "we", "they"}


def is_verb(token: str) -> bool:
    """Return True if the token is treated as a verb.

    For this simple grammar we consider any token that is not a pronoun or
    the word ``to`` as a verb. This keeps the heuristics light‑weight while
    covering the basic test cases.
    """
    return token not in PRONOUNS and token != "to"


def align_pronoun_verb(tokens: List[str]) -> List[str]:
    """Ensure a verb immediately follows a pronoun.

    If a pronoun is found and the next token is not a verb, the first verb
    appearing later in the sequence is moved so that it immediately follows
    the pronoun.
    """
    aligned = tokens[:]
    for i, tok in enumerate(aligned):
        if tok in PRONOUNS:
            for j in range(i + 1, len(aligned)):
                if is_verb(aligned[j]):
                    if j != i + 1:
                        verb = aligned.pop(j)
                        aligned.insert(i + 1, verb)
                    break
    return aligned


def tag_tokens(tokens: List[str]) -> List[Tuple[str, str]]:
    """Tag tokens with simple grammar labels.

    After aligning pronouns and verbs the function tags:
    - pronouns as ``PRON``
    - verbs as ``VERB``
    - the word ``to`` as ``TO`` with the following token tagged ``INF``
    - tokens ending with ``ing`` as ``GERUND``
    - all remaining tokens as ``NOUN``
    """
    tokens = align_pronoun_verb(tokens)
    result: List[Tuple[str, str]] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in PRONOUNS:
            result.append((tok, "PRON"))
            if i + 1 < len(tokens):
                result.append((tokens[i + 1], "VERB"))
                i += 2
                if i < len(tokens):
                    if tokens[i] == "to" and i + 1 < len(tokens):
                        result.append((tokens[i], "TO"))
                        result.append((tokens[i + 1], "INF"))
                        i += 2
                    else:
                        word = tokens[i]
                        tag = "GERUND" if word.endswith("ing") else "NOUN"
                        result.append((word, tag))
                        i += 1
            else:
                i += 1
        else:
            result.append((tok, "OTHER"))
            i += 1
    return result
