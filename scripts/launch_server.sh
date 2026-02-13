#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export HF_ENDPOINT="https://hf-mirror.com"
export PYTORCH_JIT_LOG_LEVEL='profiling_graph_executor_impl'
export HYDRA_FULL_ERROR=1
export PYTHONPATH="${SCRIPT_DIR}/../:${PYTHONPATH:-}"
export LD_LIBRARY_PATH=/data/TensorRT-10.13.0.35/lib/:/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH

python3 "${SCRIPT_DIR}/../serving/policy_server.py" "$@"