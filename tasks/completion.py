"""Text completion task."""

from src.task import Task


class Completion(Task):
    def payload(self, prompt: str, opts: dict) -> dict:
        return {
            "uri": "/completions",
            "payload": {
                "model": self.model(),
                "prompt": prompt,
                **opts,
            },
        }
