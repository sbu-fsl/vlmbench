"""LEval long-document comprehension benchmark."""

from dataloaders.leval_dataset import LEvalDataset
from src.benchmark import Benchmark
from tasks.completion import Completion


def _to_text(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple)):
        return _to_text(x[0]) if x else ""
    return str(x)


class LEvalBenchmark(Benchmark):
    """LEval: long-document comprehension with shared instruction prefix."""

    def build_input(self, entry):
        doc = _to_text(entry.get("input") or "")
        questions = entry.get("instructions", [])
        if not doc or not questions:
            return "", {}
        q = _to_text(questions[0] if isinstance(questions, list) and questions else questions)
        prompt = f"{q}\n\nContext:\n{doc}"
        opts = {"temperature": 0.0, "max_tokens": 512}
        return prompt, opts

    @classmethod
    def create(cls, model: str, cache_dir: str) -> "LEvalBenchmark":
        dataset = LEvalDataset("narrative_qa", cache_dir, limit=100)
        task = Completion(model=model)
        return cls(dataset, task)
