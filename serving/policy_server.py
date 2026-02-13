import asyncio
import http
import logging
import time
import traceback
from core.inference.inference_engine import InferenceEngine
from core.inference.factory import create_inference_engine
from utils.websocket.msgpack import pack_array, unpack_array
import websockets.asyncio.server as _server
import websockets.frames

from omegaconf import OmegaConf
from pathlib import Path
from galaxea_fm.utils.config_resolvers import register_default_resolvers

register_default_resolvers()
import toml

from utils.websocket.msgpack import Packer, unpackb

from accelerate import PartialState
distributed_state = PartialState()
from utils.torch_utils import dict_apply
import numpy as np
import torch

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        engine: InferenceEngine,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = engine
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = unpackb(await websocket.recv())

                infer_time = time.monotonic()
                obs = dict_apply(obs, lambda x: torch.from_numpy(x).cuda() if isinstance(x, np.ndarray) else x)
                action = self._policy.predict_action(obs)
                action = dict_apply(action, lambda x: x.cpu().numpy() if isinstance(x, torch.Tensor) else x)
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                print(f'infer_ms: {infer_time * 1000}')
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                send_start = time.monotonic()
                await websocket.send(packer.pack(action))
                send_cost = time.monotonic() - send_start
                prev_total_time = time.monotonic() - start_time
                # print(f'prev_total_time: {prev_total_time * 1000}, send_cost: {send_cost}')

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None

def load_config(model_path=None):
    """Load config from model_path/efmnode.toml if available, else default config.toml.

    When model_path is provided:
      - Use <model_path>/efmnode.toml if it exists
      - Otherwise fall back to default config.toml with a warning
      - Always override ckpt_dir to point to model_path
    """
    import os
    import sys
    default_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.toml")

    if model_path is not None:
        model_config_path = os.path.join(model_path, "efmnode.toml")
        if os.path.isfile(model_config_path):
            print(f"[INFO] Loading config from: {model_config_path}")
            config = toml.load(model_config_path)
        else:
            print(f"[WARNING] {model_config_path} not found, falling back to default config.toml", file=sys.stderr)
            config = toml.load(default_config_path)

        config.setdefault("model", {})
        config["model"]["ckpt_dir"] = model_path
        print(f"[INFO] Model checkpoint dir: {model_path}")
    else:
        config = toml.load(default_config_path)

    return config


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="EFMNode policy server")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Absolute path to model directory (overrides ckpt_dir in config)")
    args = parser.parse_args()

    config = load_config(args.model_path)
    config_path = config['model']['ckpt_dir']
    print(config_path)
    cfg = OmegaConf.load(f"{config['model']['ckpt_dir']}/config.yaml")
    engine = create_inference_engine(config, cfg, use_trt=config['model']['use_trt'], role="server")
    engine.load_model()
    server = WebsocketPolicyServer(engine, host=config['websocket']['host'], port=config['websocket']['port'])
    server.serve_forever()
