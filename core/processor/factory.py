from typing import Dict, Any, Optional
from omegaconf import DictConfig
from loguru import logger

from core.processor.processor import Processor
from core.processor.base_processor import BaseProcessor

def create_processor(
    config: Dict[str, Any],
    cfg: DictConfig,
    processor_type: Optional[str] = None
) -> Processor:
    """
    工厂函数：根据配置创建对应的处理器
    
    Args:
        config: 配置字典（从 config.toml 加载）
        cfg: OmegaConf 配置对象（从模型 checkpoint 加载）
        processor_type: 处理器类型，如果为 None 则使用默认的 BaseProcessor
        
    Returns:
        处理器实例
        
    Raises:
        ValueError: 如果 processor_type 不支持
    """
    # 如果没有指定类型，使用默认的 BaseProcessor
    if processor_type is None:
        processor_type = "default"
    
    processor_type = processor_type.lower()
    
    if processor_type == "default":
        logger.info("Creating base processor")
        return BaseProcessor(config=config, cfg=cfg)
    else:
        raise ValueError(f"Unsupported processor type: {processor_type}. Supported types: 'galaxea', 'default'")
