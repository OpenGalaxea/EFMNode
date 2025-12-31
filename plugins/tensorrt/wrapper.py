import torch
import ctypes
import os
import tensorrt as trt
from loguru import logger

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

class TRTWrapper:
    def __init__(self, engine_path: str, device: str, plugin_path: str = None):
        self.device = torch.device(device)
        self.engine_path = engine_path

        if plugin_path and os.path.exists(plugin_path):
            ctypes.CDLL(plugin_path, mode=ctypes.RTLD_GLOBAL)
            trt.init_libnvinfer_plugins(TRT_LOGGER, "")

        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if not self.engine:
            raise RuntimeError(f"Failed to deserialize engine: {engine_path}")

        self.context = self.engine.create_execution_context()

        self.tensor_info = {}

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            self.tensor_info[name] = {
                "mode": self.engine.get_tensor_mode(name),  # Input/Output
                "dtype": self._trt_dtype_to_torch(self.engine.get_tensor_dtype(name)),
                "shape": tuple(self.engine.get_tensor_shape(name)),
                "name": name,
            }

    def _trt_dtype_to_torch(self, trt_dtype):
        mapping = {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
            trt.bfloat16: torch.bfloat16,
            trt.int32: torch.int32,
            trt.int64: torch.int64,
            trt.bool: torch.bool,
            trt.int8: torch.int8,
        }
        return mapping.get(trt_dtype, torch.float32)

    def bind_tensor(self, name: str, tensor: torch.Tensor):
        if name not in self.tensor_info:
            return 

        expected_dtype = self.tensor_info[name]["dtype"]
        if tensor.dtype != expected_dtype:
            logger.error(
                f"[{name}] Dtype mismatch! Engine: {expected_dtype}, Got: {tensor.dtype}."
            )
        self.context.set_tensor_address(name, tensor.data_ptr())

    def execute_async(self, stream: torch.cuda.Stream):
        if not self.context.all_binding_shapes_specified:
            logger.error(
                f"Not all binding shapes specified for {os.path.basename(self.engine_path)}"
            )
        self.context.execute_async_v3(stream_handle=stream.cuda_stream)

class MemoryManager:

    def __init__(
        self, encoder: TRTWrapper, predictor: TRTWrapper, device: torch.device
    ):
        self.device = device
        self.buffers: Dict[str, torch.Tensor] = {}

        enc_names = set(encoder.tensor_info.keys())
        pred_names = set(predictor.tensor_info.keys())
        all_names = enc_names | pred_names
        shared_names = enc_names & pred_names

        logger.info(
            f"MemoryManager Analysis: {len(shared_names)} shared tensors found."
        )

        for name in all_names:
            if name in encoder.tensor_info:
                info = encoder.tensor_info[name]
            else:
                info = predictor.tensor_info[name]
            self.buffers[name] = torch.empty(
                info["shape"],
                dtype=info["dtype"],
                device=device,
                memory_format=torch.contiguous_format,
            )