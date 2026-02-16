"""HuggingFace dataset adapter."""

from typing import Optional

from src.dataset import Dataset


class HFDataset(Dataset):
    """Wraps a HuggingFace `datasets.load_dataset` call."""

    def __init__(
        self,
        name: str,
        config: Optional[str],
        split: str,
        cache_dir: str,
        limit: Optional[int] = None,
    ):
        super().__init__(name)
        self._config = config
        self._split = split
        self._cache_dir = cache_dir
        self._limit = limit
        self._ds = None
        self._iter = None
        self._n = 0

    def _load(self):
        if self._ds is None:
            import datasets as hfds

            if self._config:
                self._ds = hfds.load_dataset(
                    self._addr,
                    self._config,
                    split=self._split,
                    cache_dir=self._cache_dir,
                )
            else:
                self._ds = hfds.load_dataset(
                    self._addr, split=self._split, cache_dir=self._cache_dir
                )
            self._iter = iter(self._ds)

    def count(self) -> int:
        self._load()
        total = len(self._ds)
        return min(total, self._limit) if self._limit else total

    def next(self):
        self._load()
        if self._limit is not None and self._n >= self._limit:
            raise StopIteration
        row = next(self._iter)
        self._n += 1
        return dict(row)

