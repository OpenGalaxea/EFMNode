from typing import Dict, Any
from omegaconf import DictConfig
from loguru import logger

from core.inference.inference_engine import InferenceEngine
from core.inference.pytorch_engine import PyTorchEngine

def create_inference_engine(
    config: Dict[str, Any],
    cfg: DictConfig,
    use_trt: bool = False,
    trt_config: Dict[str, Any] = None
) -> InferenceEngine:
    logger.info("Creating PyTorch inference engine")
    return PyTorchEngine(config, cfg)


