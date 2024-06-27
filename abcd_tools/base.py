"""Base classes for download module."""

from abc import ABC, abstractmethod


class AbstractDownloader(ABC):
    """Base downloader class."""
    @abstractmethod
    def download():
        """Abstract download method."""
        raise NotImplementedError

class AbstractParser(ABC):
    """Base parser class."""
    @abstractmethod
    def parse():
        """Abstract parse method."""
        raise NotImplementedError

class AbstractReorganizer(ABC):
    """Base reorganizer class."""
    @abstractmethod
    def reorganize():
        """Abstract reorganize method."""
        raise NotImplementedError
