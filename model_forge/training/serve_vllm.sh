#!/bin/bash
# ==============================================================================
# Serve InvariantMind-v1 via vLLM with an OpenAI-compatible API
# Allows World A (evolution_sandbox) and World B (synthetic_agora) to call
# InvariantMind via standard LLM client tool calling.
# ==============================================================================

PORT=${1:-8000}
MODEL_PATH=${2:-"./checkpoints/invariant_mind_oracle_14b"}
BASE_MODEL=${3:-"deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"}

echo "Serving InvariantMind-v1 on port $PORT..."
echo "Base model: $BASE_MODEL | LoRA adapter: $MODEL_PATH"

pip install vllm

python -m vllm.entrypoints.openai.api_server \
    --model $BASE_MODEL \
    --enable-lora \
    --lora-modules invariant-mind=$MODEL_PATH \
    --port $PORT \
    --host 0.0.0.0 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code
