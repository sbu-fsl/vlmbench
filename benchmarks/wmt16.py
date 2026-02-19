"""WMT16 translation benchmark for shared prefix."""

from dataloaders.local_dataset import LocalDataset
from src.benchmark import Benchmark
from tasks.completion import Completion


class WMT16Benchmark(Benchmark):
    """WMT16: translation tasks from German to English."""

    def build_input(self, entry):
        text = entry.get("translation") # "{""de"": ""Wiederaufnahme der Sitzungsperiode"", ""en"": ""Resumption of the session""}"
        if not text or len(text.split()) < 1:
            return "", {}
        
        text = text.replace('""', '"')
        text = text.strip('"')

        import json
        try:
            text = json.loads(text).get("de", "")
        except json.JSONDecodeError:
            return "", {}
        
        prompt = f"Translate this German text to English:\n\n{text}"
        opts = {"temperature": 0.0, "max_tokens": 512}
        return prompt, opts

    @classmethod
    def create(cls, model: str, cache_dir: str) -> "WMT16Benchmark":
        dataset = LocalDataset("/mnt/gpfs/llm-datasets/wmt16.csv", limit=100)
        task = Completion(model=model)
        return cls(dataset, task)
