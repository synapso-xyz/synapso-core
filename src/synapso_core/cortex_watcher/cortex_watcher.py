from typing import List


class CortexWatcher:
    """
    A class that watches a cortex and updates the cortex when it changes.
    """

    @staticmethod
    def register_cortex(cortex_id: str) -> bool:
        return True

    @staticmethod
    def unregister_cortex(cortex_id: str) -> bool:
        return True

    @staticmethod
    def get_cortex_watchers(cortex_id: str) -> List[str]:
        return []
