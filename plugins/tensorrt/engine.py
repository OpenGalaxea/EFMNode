import torch
from loguru import logger
from omegaconf import DictConfig
from plugins.tensorrt import TRTWrapper, MemoryManager
import os
from typing import Dict
import math
import torch.nn as nn

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(
        self,
        t: torch.FloatTensor,
        max_period: float = 10000.0,
    ) -> torch.FloatTensor:
        half_dim = self.dim // 2
        emb = math.log(max_period) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=t.dtype) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class TRTInferenceEngine:
    def __init__(
        self,
        cfg: DictConfig,
        encoder_path: str,
        predictor_path: str,
        device: str = "cuda:0",
        precision: str = "fp32",
        plugin_path: str = None,
        use_cuda_graph: bool = True,
    ):
        self.device = torch.device(device)
        self.device_str = device
        self.stream = torch.cuda.Stream(device=self.device)
        self.use_cuda_graph = use_cuda_graph
        self.precision = precision

        model_cfg = cfg.model.model_arch
        self.num_inference_steps = model_cfg.num_inference_steps
        self.final_action_clip = model_cfg.get("final_action_clip_value", None)

        # load Engine
        enc_engine_file = self._get_engine(encoder_path)
        pred_engine_file = self._get_engine(predictor_path)

        # init context
        self.encoder = TRTWrapper(enc_engine_file, device, plugin_path)
        self.predictor = TRTWrapper(pred_engine_file, device, plugin_path)

        # mem allocat and manage
        self.mem = MemoryManager(self.encoder, self.predictor, self.device)
        # bind io mem and shared mem
        self._bind_memory()

        # pre-compute time cond
        if model_cfg.get("action_expert_adaptive_mode", False):
            time_embedding_dim = model_cfg.joint.time_hidden_size
        else:
            time_embedding_dim = model_cfg.joint.mixture.action.hidden_size
        self.time_embedding = SinusoidalPosEmb(time_embedding_dim).to(self.device)
        self._precompute_time_cond()

        # warmup and capture cuda graph
        self.encoder_graph = None
        self.predictor_graph = None
        if self.use_cuda_graph:
            self._warmup_and_capture()

        logger.info(f"[TRTInferenceEngine] Inference Engine Ready on {device}")

    def _get_engine(self, path: str) -> str:
        if path.endswith((".engine", ".plan")):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Engine file not found: {path}")
            return path

    def _bind_memory(self):
        for name in self.encoder.tensor_info:
            if name in self.mem.buffers:
                self.encoder.bind_tensor(name, self.mem.buffers[name])
        for name in self.predictor.tensor_info:
            if name in self.mem.buffers:
                self.predictor.bind_tensor(name, self.mem.buffers[name])
        logger.info("Memory Binding Complete.")

    def _precompute_time_cond(self):
        self.cached_time = []
        delta_t = 1.0 / self.num_inference_steps
        t_val = 0.0
        t_tensor = torch.zeros(1, dtype=torch.float32, device=self.device)
        target_dtype = self.predictor.tensor_info.get("time_cond", {}).get(
            "dtype", torch.float32
        )
        with torch.inference_mode():
            for _ in range(self.num_inference_steps):
                t_tensor.fill_(t_val)
                cond = self.time_embedding(t_tensor).clone().to(dtype=target_dtype)
                self.cached_time.append(cond)
                t_val += delta_t

    def _warmup_and_capture(self):
        logger.info("[TRTInferenceEngine] Starting Warmup and CUDA Graph Capture...")
        for name, tensor in self.mem.buffers.items():
            if "input_ids" in name:
                tensor.fill_(1)
            elif "mask" or "kv_cache" in name:
                tensor.fill_(0)
            elif tensor.is_floating_point():
                tensor.normal_()
        stream = self.stream
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.encoder.execute_async(stream)
                self.predictor.execute_async(stream)
        stream.synchronize()

        logger.info("[TRTInferenceEngine] Capturing Encoder Graph...")
        self.encoder_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.encoder_graph, stream=stream):
            self.encoder.execute_async(stream)

        logger.info("[TRTInferenceEngine] Capturing Predictor Graph...")
        self.predictor_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.predictor_graph, stream=stream):
            self.predictor.execute_async(stream)
        stream.synchronize()
        logger.info("[TRTInferenceEngine] CUDA Graph Capture Done.")

    def predict_action(self, batch: Dict[str, torch.Tensor]):
        with torch.inference_mode(), torch.cuda.stream(self.stream):
            for name, tensor in batch.items():
                if name in self.mem.buffers:
                    self.mem.buffers[name].copy_(tensor, non_blocking=True)
                    # handling the numerical safety of itp mask and action mask (avoiding NaN under FP16)
                    if "mask" in name:
                        self.mem.buffers[name].clamp_(min=-60000.0)

        # Prefill Stage
        if self.encoder_graph:
            self.encoder_graph.replay()
        else:
            self.encoder.execute_async(self.stream)

        debug_kv_cache = self.mem.buffers["kv_cache"]
        k_all = debug_kv_cache[:, 0]  # [18, 1, 1, 824, 256]
        v_all = debug_kv_cache[:, 1]  # [18, 1, 1, 824, 256]

        k_vlm = k_all.narrow(3, 0, 823)  # [18, 1, 1, 823, 256]
        v_vlm = v_all.narrow(3, 0, 823)
        k_proprio = k_all.narrow(3, 823, 1)  # [18, 1, 1, 1, 256]
        v_proprio = v_all.narrow(3, 823, 1)

        from galaxea_fm.models.kv_cache import KVCache

        kv_caches = {"vlm": KVCache(), "proprio": KVCache()}
        # torch可以对range做静态展开
        kv_caches["vlm"].key_cache = [
            k_vlm[i].clone() for i in range(18)
        ]
        kv_caches["vlm"].value_cache = [
            v_vlm[i].clone() for i in range(18)
        ]
        kv_caches["proprio"].key_cache = [
            k_proprio[i].clone() for i in range(18)
        ]
        kv_caches["proprio"].value_cache = [
            v_proprio[i].clone() for i in range(18)
        ]
        # --- Decode Loop Stage ---
        # init noisy action
        self.mem.buffers["noisy_action"].normal_()

        # --- 4. Denoising Loop ---
        for step in range(self.num_inference_steps):
            self.mem.buffers["time_cond"].copy_(
                self.cached_time[step], non_blocking=True
            )
            if self.predictor_graph:
                self.predictor_graph.replay()
            else:
                self.predictor.execute_async(self.stream)

            # Update Step: noisy_action <- denoised_action
            self.mem.buffers["noisy_action"].copy_(
                self.mem.buffers["denoised_action"], non_blocking=True
            )

        self.stream.synchronize()
        result = self.mem.buffers["denoised_action"].clone()
        if self.final_action_clip:
            result.clamp_(-self.final_action_clip, self.final_action_clip)


        return result