#!/usr/bin/env bash
# Serve Gemma 4 with llama.cpp's OpenAI-compatible server, for data generation.
#
#   scripts/serve-gemma.sh            # 12B, all layers on the GPU (the default)
#   scripts/serve-gemma.sh 26b        # 26B-A4B, attention on the GPU, experts in RAM
#   scripts/serve-gemma.sh 12b cpu    # CPU only
#
# Needs `llama-server` from llama.cpp (built with CUDA for GPU use). The model is
# downloaded from Hugging Face on first use and cached by llama.cpp.
#
# Environment:
#   LLAMA_SERVER   path to llama-server (default: llama-server on PATH)
#   MODEL          a local .gguf to serve instead of downloading one
#   PORT           port to listen on (default 8080)
#   THREADS        CPU threads (default: all cores)
#   PARALLEL       requests served at once (default: LLM_PARALLEL, or 4)
#   CONTEXT        total context tokens, shared by the parallel requests (default 16384)
#   CPU_MOE        26b on GPU: how many layers keep their experts in RAM (default: all of
#                  them, which fits any GPU; fewer is faster if VRAM allows, e.g. 20 for 12 GB)
set -euo pipefail

size=${1:-12b}
device=${2:-gpu}
server=${LLAMA_SERVER:-llama-server}
port=${PORT:-8080}
threads=${THREADS:-$(getconf _NPROCESSORS_ONLN)}
parallel=${PARALLEL:-${LLM_PARALLEL:-4}}
context=${CONTEXT:-16384}
cpu_moe=${CPU_MOE:-}

case $size in
  12b) repo=google/gemma-4-12B-it-qat-q4_0-gguf ;;
  26b) repo=google/gemma-4-26B-A4B-it-qat-q4_0-gguf ;;
  *) echo "size must be 12b or 26b" >&2; exit 2 ;;
esac

if [ -n "${MODEL:-}" ]; then
  source=(-m "$MODEL")
else
  source=(-hf "$repo")
fi

args=("${source[@]}" --host 127.0.0.1 --port "$port" -c "$context" -np "$parallel" --jinja -t "$threads")

case $device in
  gpu)
    if [ "$size" = 26b ]; then
      if [ -n "$cpu_moe" ]; then args+=(-ngl 99 --n-cpu-moe "$cpu_moe"); else args+=(-ngl 99 --cpu-moe); fi
    else
      args+=(-ngl 99)
    fi
    ;;
  cpu) export CUDA_VISIBLE_DEVICES=""; args+=(-ngl 0) ;;
  *) echo "device must be gpu or cpu" >&2; exit 2 ;;
esac

exec "$server" "${args[@]}"
