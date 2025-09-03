import asyncio
from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
from me import ChatEngine


class DummyCore:
    def __init__(self, replies):
        self.replies = replies

    async def reply(self, message: str) -> str:
        await asyncio.sleep(0)
        return self.replies.pop(0)

def test_chat_engine_punctuation_adjustment():
    async def run():
        chat = ChatEngine()
        chat.core = DummyCore(['jump.', 'run.'])
        first = await chat.reply('run!')
        assert first == 'jump.'
        second = await chat.reply('hello.')
        assert second == 'run!'

    asyncio.run(run())
