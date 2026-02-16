from abc import ABC, abstractmethod


class Dataset(ABC):
    def __init__(self, address: str):
        self._addr = address

    def address(self) -> str:
        return self._addr

    @abstractmethod
    def count(self) -> int:
        """Total number of entries."""
        pass

    @abstractmethod
    def next(self):
        """Return the next entry, or raise StopIteration."""
        pass
