"""LEval dataset adapter (manual download from HuggingFace Hub)."""

import json
from typing import Optional

from src.dataset import Dataset

_TASK_MAP = {
    "narrative_qa": "LEval/Generation/narrative_qa.jsonl",
    "natural_question": "LEval/Generation/natural_question.jsonl",
    "meeting_summ": "LEval/Generation/meeting_summ.jsonl",
    "gov_report_summ": "LEval/Generation/gov_report_summ.jsonl",
    "multidoc_qa": "LEval/Generation/multidoc_qa.jsonl",
}


class LEvalDataset(Dataset):
    """Loads LEval JSONL data from HF Hub."""

    def __init__(
        self,
        task_name: str = "narrative_qa",
        cache_dir: str = "./data",
        limit: Optional[int] = None,
    ):
        super().__init__(task_name)
        self._cache_dir = cache_dir
        self._limit = limit
        self._data = None
        self._idx = 0

    def _load(self):
        if self._data is not None:
            return
        from huggingface_hub import hf_hub_download

        filepath = _TASK_MAP.get(self._addr, "LEval/Generation/narrative_qa.jsonl")
        jsonl_path = hf_hub_download(
            repo_id="L4NLP/LEval",
            filename=filepath,
            repo_type="dataset",
            cache_dir=self._cache_dir,
        )
        self._data = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self._data.append(json.loads(line))

    def count(self) -> int:
        self._load()
        total = len(self._data)
        return min(total, self._limit) if self._limit else total

    def next(self):
        self._load()
        if self._idx >= len(self._data):
            raise StopIteration
        if self._limit is not None and self._idx >= self._limit:
            raise StopIteration
        entry = self._data[self._idx]
        self._idx += 1
        return entry

