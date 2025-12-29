from abc import ABC, abstractmethod
from typing import Dict, Any
from pathlib import Path
from omegaconf import DictConfig
import torch
from loguru import logger


class InferenceEngine(ABC):
    """推理引擎基类，定义统一的接口"""
    
    def __init__(self, config: Dict[str, Any], cfg: DictConfig):
        """
        初始化推理引擎
        
        Args:
            config: 配置字典（从 config.toml 加载）
            cfg: OmegaConf 配置对象（从模型 checkpoint 加载）
        """
        self.config = config
        self.cfg = cfg
        self.model = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    @abstractmethod
    def load_model(self) -> None:
        """
        加载模型
        
        子类需要实现具体的模型加载逻辑
        """
        pass
    
    @abstractmethod
    def predict_action(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行推理，预测动作
        
        Args:
            batch: 预处理后的批次数据
            
        Returns:
            包含预测动作的字典
        """
        pass
    
    def warmup(self, hardware: str, qpos_obs_size: int, vision_obs_size: int, action_size: int) -> None:
        """
        模型预热（可选实现）
        
        Args:
            hardware: 硬件类型 ("R1_PRO" 或 "R1_LITE")
            qpos_obs_size: 关节位置观测大小
            vision_obs_size: 视觉观测大小
            action_size: 动作大小
        """
        # 默认不进行预热，子类可以重写此方法
        logger.info("Warmup skipped (not implemented)")
        pass
    
    def to_device(self, batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        """
        将批次数据移动到指定设备
        
        Args:
            batch: 批次数据
            device: 目标设备
            
        Returns:
            移动到设备后的批次数据
        """
        from utils.torch_utils import dict_apply
        return dict_apply(batch, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
    
    def to_cpu(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        将批次数据移动到 CPU
        
        Args:
            batch: 批次数据
            
        Returns:
            移动到 CPU 后的批次数据
        """
        from utils.torch_utils import dict_apply
        return dict_apply(batch, lambda x: x.cpu() if isinstance(x, torch.Tensor) else x)

