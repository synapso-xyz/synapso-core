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
        pass
    
    @abstractmethod
    def teardown(self) -> bool:
        """
        Teardown the meta store.
        """
        pass
    
    