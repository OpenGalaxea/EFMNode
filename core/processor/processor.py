from abc import ABC, abstractmethod
from typing import Dict, Any
from pathlib import Path
from omegaconf import DictConfig


class Processor(ABC):
    """处理器基类，定义统一的数据预处理和后处理接口"""
    
    def __init__(self, config: Dict[str, Any], cfg: DictConfig):
        """
        初始化处理器
        
        Args:
            config: 配置字典（从 config.toml 加载）
            cfg: OmegaConf 配置对象（从模型 checkpoint 加载）
        """
        self.config = config
        self.cfg = cfg
    
    @abstractmethod
    def initialize(self, dataset_stats_path: Path) -> None:
        """
        初始化处理器，加载数据集统计信息等
        
        Args:
            dataset_stats_path: 数据集统计信息文件路径
        """
        pass
    
    @abstractmethod
    def preprocess(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        预处理批次数据
        
        Args:
            batch: 原始批次数据
            
        Returns:
            预处理后的批次数据
        """
        pass
    
    @abstractmethod
    def postprocess(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        后处理批次数据
        
        Args:
            batch: 模型输出的批次数据
            
        Returns:
            后处理后的批次数据
        """
        pass

