"""KV probe benchmark for prefix caching."""

from dataloaders.hf_dataset import HFDataset
from src.benchmark import Benchmark
from tasks.completion import Completion


def _to_text(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)


SHARED_SYS = "You are a concise assistant. Always answer directly."
SHARED_USER = "Common preface: Use bullet points if appropriate. Now answer:"


class KVProbeBenchmark(Benchmark):
    """KV probe: shared system/user prefix for prefix caching measurement."""

    def build_input(self, entry):
        inst = _to_text(entry.get("instruction"))
        inp = _to_text(entry.get("input"))
        q = inst if not inp else f"{inst}\n\nInput: {inp}"
        if not q:
            return "", {}
        prompt = f"{SHARED_SYS}\n{SHARED_USER}\n{q}"
        opts = {"temperature": 0.0, "max_tokens": 256}
        return prompt, opts

    @classmethod
    def create(cls, model: str, cache_dir: str) -> "KVProbeBenchmark":
        dataset = HFDataset("yahma/alpaca-cleaned", None, "train", cache_dir, limit=100)
        task = Completion(model=model)
        return cls(dataset, task)
