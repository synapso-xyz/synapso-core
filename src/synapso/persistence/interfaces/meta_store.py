from abc import ABC, abstractmethod


class MetaStore(ABC):
    """
    A store for metadata.
    """

    @abstractmethod
    def setup(self) -> bool:
        """
        Setup the meta store.
        """
        return True

    @abstractmethod
    def teardown(self) -> bool:
        """
        Teardown the meta store.
        """
        return True
