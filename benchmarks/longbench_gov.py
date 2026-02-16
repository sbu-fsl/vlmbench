"""LongBench government report summarization benchmark."""

from dataloaders.longbench_dataset import LongBenchDataset
from src.benchmark import Benchmark
from tasks.completion import Completion


def _to_text(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)


class LongBenchGovBenchmark(Benchmark):
    """LongBench: government report summarization."""

    def build_input(self, entry):
        doc = _to_text(entry.get("context") or entry.get("input"))
        if not doc:
            return "", {}
        prompt = f"Summarize the following government report succinctly:\n\n{doc}"
        opts = {"temperature": 0.7, "max_tokens": 512, "top_p": 0.95}
        return prompt, opts

    @classmethod
    def create(cls, model: str, cache_dir: str) -> "LongBenchGovBenchmark":
        dataset = LongBenchDataset("gov_report", cache_dir, limit=100)
        task = Completion(model=model)
        return cls(dataset, task)
