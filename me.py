import random
from typing import List, Tuple, Set, Optional
from memory import init_db, tokenize, get_vocab, metrics
from rag import retrieve
from method import train


class Engine:
    def __init__(self) -> None:
        init_db()

    def _lengths(self, entropy: float, perplexity: float) -> Tuple[int, int]:
        base1 = 5 + int(entropy) % 5
        base2 = 5 + int(perplexity) % 5
        if base1 == base2:
            base2 = 5 + ((base2 + 1) % 5)
        return base1, base2

    def _invert_pronouns(self, words: List[str]) -> List[str]:
        # lightweight me2me: temporary perspective flip (you↔i, your↔my, etc.)
        mapping = {
            'you': 'i', 'u': 'i', 'your': 'my', 'yours': 'mine', 'yourself': 'myself',
            'i': 'you', 'me': 'you', 'my': 'your', 'mine': 'yours', 'myself': 'yourself',
            'we': 'you'
        }
        return [mapping.get(w, w) for w in words]

    def _generate(self, words: List[str], distance: float, length: int, used: Set[str], pref: Optional[List[str]] = None) -> str:
        vocab = get_vocab()
        _, _, resonance = metrics(words, vocab)
        pronouns = {'you': 'i', 'u': 'i', 'i': 'you', 'me': 'you', 'we': 'you'}
        pronoun = next((pronouns[w] for w in words if w in pronouns), None)
        candidates = retrieve(words, distance=distance)
        random.shuffle(candidates)
        sent: List[str] = []
        if pronoun and pronoun not in used:
            sent.append(pronoun)
            used.add(pronoun)
        # ephemeral me2me preference: try flipped-perspective tokens first
        if pref:
            for w in pref:
                if len(sent) >= length:
                    break
                if w in words or w in used:
                    continue
                if len(w) == 1:
                    continue
                sent.append(w)
                used.add(w)
        for w in candidates:
            if len(sent) >= length:
                break
            if w in words or w in used:
                continue
            if w == 'the' and len(sent) > 0:
                w = 'the'
            sent.append(w)
            used.add(w)
        while len(sent) < length:
            choice = random.choice(candidates) if candidates else 'hmm'
            if choice not in sent and choice not in words and choice not in used:
                sent.append(choice)
                used.add(choice)
        if len(sent[-1]) == 1:
            sent[-1] = 'hmm'
        if sent:
            sent[0] = sent[0].capitalize()
        return ' '.join(sent) + '.'

    def reply(self, message: str) -> str:
        words = tokenize(message)
        vocab = get_vocab()
        entropy, perplexity, _ = metrics(words, vocab)
        len1, len2 = self._lengths(entropy, perplexity)
        used: Set[str] = set()
        pref = self._invert_pronouns(words)
        first = self._generate(words, 0.5, len1, used, pref=pref)
        second = self._generate(words, 0.7, len2, used, pref=pref)
        train(message)
        return f"{first} {second}"


if __name__ == '__main__':
    bot = Engine()
    try:
        while True:
            msg = input('> ')
            print(bot.reply(msg))
    except KeyboardInterrupt:
        pass
