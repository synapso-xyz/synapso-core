from .cortex import Cortex
from typing import List

class CortexManager:
    def __init__(self):
        pass
    
    def get_cortex_by_id(self, id: str) -> Cortex:
        pass
    
    def create_cortex(self, name: str) -> Cortex:
        pass
    
    def delete_cortex(self, id: str) -> bool:
        pass
    
    def set_active_cortex(self, id: str) -> bool:
        pass
    
    def get_active_cortex(self) -> Cortex:
        pass
    
    def get_all_cortices(self) -> List[Cortex]:
        pass

    def index_cortex(self, id: str) -> bool:
        pass
    
    
    