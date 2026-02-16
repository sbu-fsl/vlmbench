"""Chat completion task."""

from src.task import Task


class ChatBot(Task):
    def payload(self, prompt: str, opts: dict) -> dict:
        return {
            "uri": "/chat/completions",
            "payload": {
                "model": self.model(),
                "messages": [{"role": "user", "content": prompt}],
                **opts,
            },
        }
