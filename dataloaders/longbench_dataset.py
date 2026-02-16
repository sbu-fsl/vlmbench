"""LongBench dataset adapter (manual download from HuggingFace Hub)."""

import json
import os
import zipfile
from typing import Optional

from src.dataset import Dataset


class LongBenchDataset(Dataset):
    """Loads LongBench JSONL data from HF Hub zip archive."""

    def __init__(
        self, task_name: str, cache_dir: str, limit: Optional[int] = None
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

        extract_dir = os.path.join(self._cache_dir, "longbench_extracted")
        data_file = os.path.join(extract_dir, "data", f"{self._addr}.jsonl")

        if not os.path.exists(data_file):
            os.makedirs(extract_dir, exist_ok=True)
            zip_path = hf_hub_download(
                repo_id="THUDM/LongBench",
                filename="data.zip",
                repo_type="dataset",
                cache_dir=self._cache_dir,
            )
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)

        if not os.path.exists(data_file):
            raise FileNotFoundError(f"LongBench task '{self._addr}' not found.")

        self._data = []
        with open(data_file, "r", encoding="utf-8") as f:
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

