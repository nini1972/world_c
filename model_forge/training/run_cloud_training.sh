#!/bin/bash
# ==============================================================================
# InvariantMind-v1 Cloud GPU One-Click Setup & Execution Script
# Suitable for RunPod, Lambda Labs, Google Colab Pro, or AWS EC2 (A10G, A100, H100)
# ==============================================================================

set -e

echo "===================================================================="
echo "⚡ Initializing InvariantMind-v1 Cloud Training Environment"
echo "===================================================================="

# Check GPU availability
nvidia-smi

# 1. Install Dependencies
echo "Ensuring Hugging Face transformers, PEFT, TRL, bitsandbytes are installed..."
pip install --upgrade transformers datasets accelerate peft trl bitsandbytes pyyaml tensorboard scipy matplotlib

# 2. Verify Dataset
if [ ! -f "../../data/colony_training_set_v2_full.jsonl" ]; then
    echo "Dataset not found in default path! Generating from colony archives..."
    python -c "from model_forge.harvester import ColonyDataHarvester; from model_forge.dataset_curator import DatasetCurator; h = ColonyDataHarvester(); c = DatasetCurator('../../data'); eps = h.harvest_all(); c.export_jsonl(c.curate_episodes(eps), 'colony_training_set_v2_full.jsonl')"
fi

echo "Dataset confirmed: $(wc -l < ../../data/colony_training_set_v2_full.jsonl) lines."

# 3. Train Oracle Model (DeepSeek-R1-Distill-Qwen-14B)
echo "--------------------------------------------------------------------"
echo "🚀 Training Tier 1: Oracle Reasoner (DeepSeek-R1-Distill-Qwen-14B)"
echo "--------------------------------------------------------------------"
if [ -f "./checkpoints/invariant_mind_oracle_14b/adapter_model.safetensors" ]; then
    echo "✅ SFT Adapter checkpoint already exists in ./checkpoints/invariant_mind_oracle_14b!"
    echo "Preserving converged weights and proceeding directly to alignment."
else
    python train_sft.py --tier oracle --config config.yaml
fi

# 4. Optional DPO Alignment
if [ -f "../../data/colony_dpo_pairs_v1.jsonl" ]; then
    echo "--------------------------------------------------------------------"
    echo "🎯 Running DPO Alignment on Verified vs Refuted Debates"
    echo "--------------------------------------------------------------------"
    python train_dpo.py --adapter_path ./checkpoints/invariant_mind_oracle_14b --base_model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
fi

echo "===================================================================="
echo "🎉 InvariantMind-v1 Cloud Training Run Complete!"
echo "Checkpoints saved in ./checkpoints/"
echo "===================================================================="
