#!/bin/bash
# ==============================================================================
# InvariantMind-Worker-7B Training Script
# Fine-tunes Qwen2.5-Coder-7B-Instruct on colony scientific & simulation episodes
# ==============================================================================

set -e

echo "===================================================================="
echo "⚡ Initializing InvariantMind-Worker-7B Training Environment"
echo "===================================================================="

nvidia-smi

echo "--------------------------------------------------------------------"
echo "🚀 Training Tier 2: Worker Engineer (Qwen/Qwen2.5-Coder-7B-Instruct)"
echo "--------------------------------------------------------------------"

python train_sft.py --tier worker --config config.yaml

echo "===================================================================="
echo "🎉 InvariantMind-Worker-7B Training Complete!"
echo "Checkpoint saved in ./checkpoints/invariant_mind_worker_7b"
echo "===================================================================="
