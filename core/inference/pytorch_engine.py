from pathlib import Path
from typing import Dict, Any
from omegaconf import DictConfig
import torch
from loguru import logger
from hydra.utils import instantiate

from core.inference.inference_engine import InferenceEngine
from galaxea_fm.utils.config_resolvers import register_default_resolvers

register_default_resolvers()


class PyTorchEngine(InferenceEngine):
    def load_model(self) -> None:
        logger.info("Use PyTorch ckpt")
        model = instantiate(self.cfg.model.model_arch)
        
        checkpoint_path = Path(self.config['model']['ckpt_dir'])
        state_dict_path = checkpoint_path / "model_state_dict.pt"
        
        logger.info(f"Loading state dict from {state_dict_path}")
        state_dict = torch.load(state_dict_path, map_location=self.device)
        
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            model.load_state_dict(state_dict["model_state_dict"], strict=True)
        else:
            model.load_state_dict(state_dict, strict=True)
            
        del state_dict
        
        self.model = model.to(self.device)
        if self.config['model']['is_torch_compile']:
            self.model = torch.compile(self.model, mode="max-autotune")
            self.warmup()
        self.model.eval()
        logger.info("PyTorch model loaded successfully")

    def warmup(self) -> None:
        batch = {}
        batch_size = 1
        batch["pixel_values"] = torch.randn(
            batch_size,
            self.cfg.model.model_arch.num_input_images,
            self.cfg.model.model_arch.vision.num_channels,
            self.cfg.model.model_arch.vision.image_size,
            self.cfg.model.model_arch.vision.image_size,
            dtype=torch.float32).to(self.device)

        batch['proprio'] = torch.randn(
            batch_size,
            self.cfg.model.model_arch.proprio_dim,
            dtype=torch.float32).unsqueeze(0).to(self.device)

        batch["input_ids"] = torch.randint(
            0,
            self.cfg.model.model_arch.vocab_size,
            (batch_size, self.cfg.model.model_arch.max_image_text_tokens),
            dtype=torch.long).to(self.device)
        batch["attention_mask"] = torch.ones_like(batch["input_ids"], dtype=torch.bool).to(self.device)
        batch["task"] = ["null"] * batch_size
        
        self.predict_action(batch)
        logger.info("PyTorch model warmed up successfully")
        actions = self.predict_action(batch)
    
    def predict_action(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        batch = self.to_device(batch, self.device)
        
        with torch.no_grad():
            batch = self.model.forward(batch, inference_mode=True)
        
        return batch

