import asyncio
import random
import re
from typing import List, Tuple, Set, Optional
from memory import tokenize, get_vocab, metrics, VerbGraph
from rag import retrieve
from method import train


class Engine:
    def __init__(self) -> None:
        """Initialize the engine."""

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

    def _generate(self, words: List[str], candidates: List[str], length: int, used: Set[str], pref: Optional[List[str]] = None) -> str:
        vocab = get_vocab()
        _, _, resonance = metrics(words, vocab)
        pronouns = {'you': 'i', 'u': 'i', 'i': 'you', 'me': 'you', 'we': 'you'}
        pronoun = next((pronouns[w] for w in words if w in pronouns), None)
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

    async def reply(self, message: str) -> str:
        words = tokenize(message)
        vocab = get_vocab()

        async def generate() -> str:
            metrics_future = asyncio.create_task(asyncio.to_thread(metrics, words, vocab))
            retrieve_05_future = asyncio.create_task(asyncio.to_thread(retrieve, words, distance=0.5))
            retrieve_07_future = asyncio.create_task(asyncio.to_thread(retrieve, words, distance=0.7))

            entropy, perplexity, _ = await metrics_future
            candidates_05, candidates_07 = await asyncio.gather(
                retrieve_05_future, retrieve_07_future
            )
            len1, len2 = self._lengths(entropy, perplexity)
            used: Set[str] = set()
            pref = self._invert_pronouns(words)
            first, second = await asyncio.gather(
                asyncio.to_thread(self._generate, words, candidates_05, len1, used, pref=pref),
                asyncio.to_thread(self._generate, words, candidates_07, len2, used, pref=pref),
            )
            return f"{first} {second}"

        reply_text, _ = await asyncio.gather(
            generate(), train(message)
        )
        return reply_text


class ChatEngine:
    """Dialogue engine with verb-punctuation tracking."""

    def __init__(self) -> None:
        self.core = Engine()
        self.graph = VerbGraph()

    async def reply(self, message: str) -> str:
        """Generate a reply and update verb graph."""
        self.graph.add_sentence(message)
        text = await self.core.reply(message)
        match = re.search(r"(\b\w+\b)[.!?]$", text.strip())
        if match:
            verb = match.group(1)
            punct = self.graph.preferred_punct(verb)
            text = re.sub(r"[.!?]$", punct, text.strip())
        self.graph.add_sentence(text)
        return text


if __name__ == '__main__':
    bot = ChatEngine()
    try:
        while True:
            msg = input('> ')
            print(asyncio.run(bot.reply(msg)))
    except KeyboardInterrupt:
        pass
