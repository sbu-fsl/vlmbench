"""Local dataset reader (from existing CSV file)."""

import csv
import os
from typing import Any, Dict, List, Optional

from src.dataset import Dataset


class LocalDataset(Dataset):
    """LocalDataset reads data from local CSV file."""

    def __init__(self, path: str, cache_dir: str, limit: Optional[int] = None):
        super().__init__(os.path.join(cache_dir, path))
        self._cache_dir = cache_dir
        self._limit = limit
        self._data: Optional[List[Dict[str, Any]]] = None
        self._idx = 0

    def _load(self):
        """Load CSV file into memory (lazy loading)."""
        if self._data is not None:
            return

        self._data = []
        with open(self.address(), "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if self._limit is not None and i >= self._limit:
                    break
                self._data.append(row)

    def count(self) -> int:
        self._load()
        total = len(self._data)
        return total

    def next(self):
        """Return next row or stop iteration if exhausted."""
        self._load()

        if self._idx >= len(self._data):
            raise StopIteration

        item = self._data[self._idx]
        self._idx += 1
        return item

    def reset(self):
        """Reset iteration index."""
        self._idx = 0
