from typing import List

class CortexWatcher:
    """
    A class that watches a cortex and updates the cortex when it changes.
    """

    @staticmethod
    def register_cortex(cortex_id: str) -> bool:
        pass
    
    @staticmethod
    def unregister_cortex(cortex_id: str) -> bool:
        pass
    
    @staticmethod
    def get_cortex_watchers(cortex_id: str) -> List[str]:
        pass