"""ShareGPT conversation benchmark."""

from dataloaders.sharegpt_dataset import ShareGPTDataset
from src.benchmark import Benchmark
from tasks.completion import Completion


def _to_text(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)


class ShareGPTBenchmark(Benchmark):
    """ShareGPT: conversational prompt/response pairs."""

    def build_input(self, entry):
        conv = entry.get("conversations") or entry.get("messages")
        if not conv or not isinstance(conv, list):
            return "", {}
        q = ""
        for m in conv:
            if m.get("from") in ("human", "user") or m.get("role") == "user":
                q = _to_text(m.get("value") or m.get("content"))
                break
        if not q:
            return "", {}
        prompt = q
        opts = {"temperature": 0.7, "max_tokens": 512, "top_p": 0.95}
        return prompt, opts

    @classmethod
    def create(cls, model: str, cache_dir: str) -> "ShareGPTBenchmark":
        dataset = ShareGPTDataset(cache_dir, limit=100)
        task = Completion(model=model)
        return cls(dataset, task)
