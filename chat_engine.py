import asyncio
import re

from me import Engine
from verb_graph import VerbGraph


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


if __name__ == "__main__":
    bot = ChatEngine()
    try:
        while True:
            msg = input('> ')
            print(asyncio.run(bot.reply(msg)))
    except KeyboardInterrupt:
        pass
