"""ShareGPT dataset adapter (manual download from HuggingFace Hub)."""

import json
from typing import Optional

from src.dataset import Dataset


class ShareGPTDataset(Dataset):
    """Loads ShareGPT conversations from HF Hub JSON file."""

    def __init__(self, cache_dir: str, limit: Optional[int] = None):
        super().__init__("anon8231489123/ShareGPT_Vicuna_unfiltered")
        self._cache_dir = cache_dir
        self._limit = limit
        self._data = None
        self._idx = 0

    def _load(self):
        if self._data is not None:
            return
        from huggingface_hub import hf_hub_download

        json_path = hf_hub_download(
            repo_id="anon8231489123/ShareGPT_Vicuna_unfiltered",
            filename="ShareGPT_V3_unfiltered_cleaned_split.json",
            repo_type="dataset",
            cache_dir=self._cache_dir,
        )
        with open(json_path, "r", encoding="utf-8") as f:
            self._data = json.load(f)

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

