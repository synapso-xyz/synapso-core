from typing import List

from .cortex import Cortex


class CortexManager:
    def __init__(self):
        """Initialize the cortex manager."""
        pass

    def get_cortex_by_id(self, id: str) -> Cortex | None:
        """Retrieve a cortex instance by its unique identifier."""
        pass

    def create_cortex(self, name: str) -> Cortex | None:
        """Create a new cortex with the specified name."""
        pass

    def delete_cortex(self, id: str) -> bool | None:
        """Delete a cortex by its ID. Returns True if successful."""
        pass

    def set_active_cortex(self, id: str) -> bool | None:
        """Set the specified cortex as the active one. Returns True if successful."""
        pass

    def get_active_cortex(self) -> Cortex | None:
        """Get the currently active cortex instance."""
        pass

    def get_all_cortices(self) -> List[Cortex] | None:
        """Retrieve all managed cortex instances."""
        pass

    def index_cortex(self, id: str) -> bool | None:
        """Index the specified cortex. Returns True if successful."""
        pass
