"""Base classes for abcd-tools."""

from abc import ABC, abstractmethod


class AbstractDataset(ABC):
    """Base dataset class."""
    @abstractmethod
    def load():
        """Abstract loading method."""
        raise NotImplementedError
