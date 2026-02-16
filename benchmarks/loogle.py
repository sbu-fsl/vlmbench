"""LooGLE legal text summarization benchmark."""

from dataloaders.hf_dataset import HFDataset
from src.benchmark import Benchmark
from tasks.completion import Completion


def _to_text(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        for k in ("text", "value", "answer", "content", "document", "summary", "target"):
            if k in x:
                return _to_text(x[k])
    if isinstance(x, (list, tuple)):
        for y in x:
            t = _to_text(y)
            if t:
                return t
        return ""
    return str(x)


class LooGLEBenchmark(Benchmark):
    """LooGLE: legal text summarization."""

    def build_input(self, entry):
        doc = _to_text(
            entry.get("document")
            or entry.get("context")
            or entry.get("text")
            or entry.get("passage")
        )
        if not doc:
            return "", {}
        prompt = f"Summarize the following legal text concisely:\n\n{doc}"
        opts = {"temperature": 0.7, "max_tokens": 512, "top_p": 0.95}
        return prompt, opts

    @classmethod
    def create(cls, model: str, cache_dir: str) -> "LooGLEBenchmark":
        dataset = HFDataset(
            "bigai-nlco/LooGLE", "summarization", "test", cache_dir, limit=100
        )
        task = Completion(model=model)
        return cls(dataset, task)
