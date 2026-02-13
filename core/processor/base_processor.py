from pathlib import Path
from typing import Dict, Any
from omegaconf import DictConfig
from loguru import logger
from hydra.utils import instantiate

from galaxea_fm.utils.normalizer import load_dataset_stats_from_json
from galaxea_fm.processors.base_processor import BaseProcessor as GalaxeaBaseProcessor
import torch
from core.processor.processor import Processor


class BaseProcessor(Processor):
    def __init__(self, config: Dict[str, Any], cfg: DictConfig):
        super().__init__(config, cfg)
        self.processor: GalaxeaBaseProcessor = None
    
    def initialize(self, dataset_stats_path: Path) -> None:
        logger.info(f"Initializing Galaxea processor with dataset stats from {dataset_stats_path}")
        
        # 实例化处理器
        self.processor = instantiate(self.cfg.data.processor)
        
        # 加载数据集统计信息
        dataset_stats = load_dataset_stats_from_json(dataset_stats_path)
        self.processor.set_normalizer_from_stats(dataset_stats)
        
        # 设置为评估模式
        self.processor.eval()
        
        logger.info("Galaxea processor initialized successfully")
    
    def preprocess(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if self.processor is None:
            raise RuntimeError("Processor not initialized. Call initialize() first.")
        
        return self.processor.preprocess(batch)
    
    def postprocess(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if self.processor is None:
            raise RuntimeError("Processor not initialized. Call initialize() first.")
        
        return self.processor.postprocess(batch)

