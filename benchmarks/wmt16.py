"""WMT16 translation benchmark for shared prefix."""

from dataloaders.local_dataset import LocalDataset
from src.benchmark import Benchmark
from tasks.completion import Completion


class WMT16Benchmark(Benchmark):
    """WMT16: translation tasks from German to English."""

    def build_input(self, entry):
        text = entry.get("cs", "")
        if not text or len(text.split()) < 1:
            return "", {}
        prompt = f"Translate this German text to English:\n\n{text}"
        opts = {"temperature": 0.0, "max_tokens": 512}
        return prompt, opts

    @classmethod
    def create(cls, model: str, _: str) -> "WMT16Benchmark":
        dataset = LocalDataset("wmt16.csv", limit=100)
        task = Completion(model=model)
        return cls(dataset, task)
