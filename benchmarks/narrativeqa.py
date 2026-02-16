"""NarrativeQA reading comprehension benchmark."""

from dataloaders.hf_dataset import HFDataset
from src.benchmark import Benchmark
from tasks.completion import Completion


def _to_text(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        for k in ("text", "value", "answer", "content"):
            if k in x:
                return _to_text(x[k])
    if isinstance(x, (list, tuple)):
        for y in x:
            t = _to_text(y)
            if t:
                return t
        return ""
    return str(x)


class NarrativeQABenchmark(Benchmark):
    """NarrativeQA: abstractive question answering."""

    def build_input(self, entry):
        q = _to_text(entry.get("question"))
        if not q:
            return "", {}
        prompt = (
            "Provide a concise answer (a few words) to the question. No explanation.\n\n"
            f"Question: {q}\nFinal answer:"
        )
        opts = {"temperature": 0.0, "max_tokens": 256}
        return prompt, opts

    @classmethod
    def create(cls, model: str, cache_dir: str) -> "NarrativeQABenchmark":
        dataset = HFDataset("narrativeqa", None, "validation", cache_dir, limit=100)
        task = Completion(model=model)
        return cls(dataset, task)
