"""HumanEval code completion benchmark."""

from dataloaders.hf_dataset import HFDataset
from src.benchmark import Benchmark
from tasks.completion import Completion


class HumanEvalBenchmark(Benchmark):
    """HumanEval: Python code completion."""

    def build_input(self, entry):
        prompt = entry.get("prompt", "")
        if not prompt:
            return "", {}
        opts = {"temperature": 0.2, "max_tokens": 256, "top_p": 0.95}
        return prompt, opts

    @classmethod
    def create(cls, model: str, cache_dir: str) -> "HumanEvalBenchmark":
        dataset = HFDataset("openai_humaneval", None, "test", cache_dir, limit=164)
        task = Completion(model=model)
        return cls(dataset, task)
