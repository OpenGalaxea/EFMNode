from abc import ABC, abstractmethod
from typing import Dict, Any
from pathlib import Path
from omegaconf import DictConfig


class Processor(ABC):
    def __init__(self, config: Dict[str, Any], cfg: DictConfig):
        self.config = config
        self.cfg = cfg
    
    @abstractmethod
    def initialize(self, dataset_stats_path: Path) -> None:
        pass
    
    @abstractmethod
    def preprocess(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def postprocess(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        pass

