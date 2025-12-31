from typing import Dict, Any
from omegaconf import DictConfig
from loguru import logger

from core.inference.inference_engine import InferenceEngine

class TensorRTEngine(InferenceEngine):
    def __init__(self, config: Dict[str, Any], cfg: DictConfig, 
                 encoder_path: str = None,
                 predictor_path: str = None,
                 device: str = "cuda:0",
                 precision: str = "fp16",
                 plugin_path: str = None,
                 use_cuda_graph: bool = True):
        super().__init__(config, cfg)
        self.encoder_path = encoder_path
        self.predictor_path = predictor_path
        self.device_str = device
        self.precision = precision
        self.plugin_path = plugin_path
        self.use_cuda_graph = use_cuda_graph
    
    def load_model(self) -> None:
        from plugins.tensorrt import TRTInferenceEngine
        logger.info("Use TensorRT")
        
        if self.encoder_path is None or self.predictor_path is None:
            raise ValueError("encoder_path and predictor_path must be provided for TensorRT engine")
        
        self.model = TRTInferenceEngine(
            cfg=self.cfg,
            encoder_path=self.encoder_path,
            predictor_path=self.predictor_path,
            device=self.device_str,
            precision=self.precision,
            plugin_path=self.plugin_path,
            use_cuda_graph=self.use_cuda_graph,
        )
        logger.info("TensorRT model loaded successfully")
    
    def predict_action(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        batch["action"] = self.model.predict_action(batch)
        return batch

