#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export HF_ENDPOINT="https://hf-mirror.com"
export PYTORCH_JIT_LOG_LEVEL='profiling_graph_executor_impl'
export HYDRA_FULL_ERROR=1
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30 python3 "${SCRIPT_DIR}/run.py"
