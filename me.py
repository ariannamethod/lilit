import random
from typing import List, Tuple, Set
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

    def _generate(self, words: List[str], distance: float, length: int, used: Set[str]) -> str:
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
        sent[0] = sent[0].capitalize()
        return ' '.join(sent) + '.'

    def reply(self, message: str) -> str:
        words = tokenize(message)
        vocab = get_vocab()
        entropy, perplexity, _ = metrics(words, vocab)
        len1, len2 = self._lengths(entropy, perplexity)
        used: Set[str] = set()
        first = self._generate(words, 0.5, len1, used)
        second = self._generate(words, 0.7, len2, used)
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
