"""Wikitext language modeling benchmark."""

from dataloaders.hf_dataset import HFDataset
from src.benchmark import Benchmark
from tasks.completion import Completion


class WikitextBenchmark(Benchmark):
    """Wikitext: text continuation from first half of a passage."""

    def build_input(self, entry):
        text = entry.get("text", "")
        if not text or len(text.split()) < 20:
            return "", {}
        words = text.split()
        mid = len(words) // 2
        prompt_text = " ".join(words[:mid])
        prompt = f"Continue the following text:\n\n{prompt_text}"
        opts = {"temperature": 0.0, "max_tokens": 512}
        return prompt, opts

    @classmethod
    def create(cls, model: str, cache_dir: str) -> "WikitextBenchmark":
        dataset = HFDataset(
            "wikitext", "wikitext-2-raw-v1", "test", cache_dir, limit=100
        )
        task = Completion(model=model)
        return cls(dataset, task)
