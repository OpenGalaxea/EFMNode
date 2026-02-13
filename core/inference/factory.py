import os
from typing import Dict, Any, Optional
from omegaconf import DictConfig
from loguru import logger

from core.inference.inference_engine import InferenceEngine
from core.inference.pytorch_engine import PyTorchEngine
from core.inference.tensorrt_engine import TensorRTEngine
from core.inference.websocket_engine import WebSocketClientEngine

def create_inference_engine(
    config: Dict[str, Any],
    cfg: DictConfig,
    use_trt: bool = False,
    trt_config: Dict[str, Any] = None,
    role: Optional[str] = None
) -> InferenceEngine:
    if use_trt:
        logger.info("Creating TensorRT inference engine")
        default_trt_config = {}
        ckpt_path = config["model"]["ckpt_dir"]
        default_trt_config["encoder_path"] = os.path.join(ckpt_path, "galaxea_zero_encoder_opt.fp16.engine")
        default_trt_config["predictor_path"] = os.path.join(ckpt_path, "galaxea_zero_predictor_opt.fp16.engine")
        default_trt_config["device"] = "cuda:0"
        default_trt_config["precision"] = "fp16"
        default_trt_config["plugin_path"] = os.path.join(ckpt_path, "gemma_rmsnorm.so")
        default_trt_config["use_cuda_graph"] = True
        if trt_config is None:
            trt_config = default_trt_config
        return TensorRTEngine(
            config=config,
            cfg=cfg,
            encoder_path=trt_config.get("encoder_path"),
            predictor_path=trt_config.get("predictor_path"),
            device=trt_config.get("device", "cuda:0"),
            precision=trt_config.get("precision", "fp16"),
            plugin_path=trt_config.get("plugin_path"),
            use_cuda_graph=trt_config.get("use_cuda_graph", True),
        )
    elif use_trt is False:
        logger.info("Creating PyTorch inference engine")
        return PyTorchEngine(config, cfg)


