from core.processor.processor import Processor
from galaxea_fm.utils.normalizer import load_dataset_stats_from_json
from galaxea_fm.processors.base_processor import BaseProcessor as GalaxeaBaseProcessor
from typing import Dict, Any
from omegaconf import DictConfig
from pathlib import Path
from loguru import logger
import torch
from hydra.utils import instantiate

class CUDAProcessor(Processor):
    def __init__(self, config: Dict[str, Any], cfg: DictConfig):
        self.cfg = cfg
        self.processor: GalaxeaBaseProcessor = None
        self.num_proprio_tokens = cfg.model.processor.num_obs_steps
        self.num_action_tokens = cfg.model.model_arch.horizon_steps
        self.max_image_text_tokens = cfg.model.model_arch.max_image_text_tokens
        self.total_num_tokens = (
            cfg.model.model_arch.max_image_text_tokens
            + self.num_proprio_tokens
            + self.num_action_tokens
        )

    def initialize(self, dataset_stats_path: Path) -> None:
        """
        初始化处理器，加载数据集统计信息
        
        Args:
            dataset_stats_path: 数据集统计信息文件路径
        """
        logger.info(f"Initializing Galaxea processor with dataset stats from {dataset_stats_path}")

        self.processor = instantiate(self.cfg.model.processor)

        dataset_stats = load_dataset_stats_from_json(dataset_stats_path)
        self.processor.set_normalizer_from_stats(dataset_stats)

        self.processor.eval()

        logger.info("Galaxea processor initialized successfully")
    
    def preprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        sample = self.processor.preprocess(data)
        """
        block attention --- padding for unused text tokens

                 img/text img/text img/text (padding) proprio action action
        img/text    x        x        x
        img/text    x        x        x
        img/text    x        x        x
        (padding)
        proprio     x        x        x                 x
        action      x        x        x                 x       x      x
        action      x        x        x                 x       x      x
        """
        attention_mask_1d = sample["attention_mask"]
        device = attention_mask_1d.device

        dtype = torch.float32
        proprio_start = self.max_image_text_tokens
        proprio_end = self.max_image_text_tokens + self.num_proprio_tokens
        action_start = proprio_end

        cnt = torch.sum(attention_mask_1d).long().item()
        causal_mask = torch.full(
            (self.total_num_tokens, self.total_num_tokens),
            torch.finfo(dtype).min,  # 极小值
            dtype=dtype,
            device=device,
        )

        causal_mask[:cnt, :cnt] = 0
        causal_mask[proprio_start:, :cnt] = 0
        causal_mask[proprio_start:proprio_end, proprio_start:proprio_end] = 0
        causal_mask[action_start:, proprio_start:] = 0
        causal_mask[action_start:, action_start:] = 0
        causal_mask = causal_mask.unsqueeze(0)

        split_idx = self.max_image_text_tokens + self.num_proprio_tokens

        sample["image_text_proprio_mask"] = causal_mask[:, :split_idx, :split_idx]
        sample["action_mask"] = causal_mask[:, -self.num_action_tokens :, :]

        return sample

    def postprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return self.processor.postprocess(data)
