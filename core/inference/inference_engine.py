from abc import ABC, abstractmethod
from typing import Dict, Any
from pathlib import Path
from omegaconf import DictConfig
import torch
from loguru import logger


class InferenceEngine(ABC):
    def __init__(self, config: Dict[str, Any], cfg: DictConfig):
        self.config = config
        self.cfg = cfg
        self.model = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    @abstractmethod
    def load_model(self) -> None:
        pass
    
    @abstractmethod
    def predict_action(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        pass
    
    def warmup(self, hardware: str, qpos_obs_size: int, vision_obs_size: int, action_size: int) -> None:
        logger.info("Warmup skipped (not implemented)")
        pass
    
    def to_device(self, batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        from utils.torch_utils import dict_apply
        return dict_apply(batch, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
    
    def to_cpu(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        from utils.torch_utils import dict_apply
        return dict_apply(batch, lambda x: x.cpu() if isinstance(x, torch.Tensor) else x)

