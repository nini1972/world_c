# 🧠 InvariantMind-v1 Cloud Training Playbook

This playbook provides complete, end-to-end instructions for fine-tuning **InvariantMind-v1** on cloud GPUs using the colony's 19.4 MB curated dataset (`data/colony_training_set_v2_full.jsonl`) and preference pairs (`data/colony_dpo_pairs_v1.jsonl`).

---

## 1. Two-Tier Neural Cognitive Architecture

Following the colony's convocation and architectural recommendations:
1. **Tier 1 — The Oracle / Synthesizer (`DeepSeek-R1-Distill-Qwen-14B`):**
   - **Role:** High-level mathematical proofs, phase-space invariant extraction, cross-world treaty synthesis, and epistemic self-correction.
   - **Cognitive Style:** Deep chain-of-thought `<think>` traces, zero generative ego, strict adherence to empirical data over narrative.
2. **Tier 2 — The Worker / Tool-Caller (`Qwen/Qwen2.5-Coder-7B-Instruct`):**
   - **Role:** High-throughput numerical simulation scripting, `colony_lib` API calls, parameter sweeps, and headless Matplotlib generation.
   - **Cognitive Style:** Fast, deterministic, zero syntax errors, automated error recovery.

---

## 2. Cloud GPU Recommendation & Cost

| Platform | Recommended GPU | VRAM | Training Time (3 Epochs) | Estimated Cost |
| :--- | :---: | :---: | :---: | :---: |
| **RunPod.io** (Recommended) | 1x NVIDIA A100 (SXM / PCIe) | 80 GB | ~45 minutes | **~$1.60 – $2.00** |
| **RunPod.io / Lambda** | 1x NVIDIA A10G / L40S | 24 GB / 48 GB | ~75 minutes | **~$1.20 – $1.80** |
| **Google Colab Pro** | 1x NVIDIA A100 | 40 GB | ~60 minutes | **10–15 Compute Units** |

---

## 3. Step-by-Step Cloud Execution (e.g. RunPod / Lambda / Colab)

### Step 1: Start GPU Pod / Instance
1. Launch an instance with **PyTorch 2.4+ / CUDA 12.1+** image (e.g., `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`).
2. Open the Web Terminal or SSH into the instance.

### Step 2: Clone the Repository & Sync Dataset
```bash
git clone https://github.com/nini1972/world_c.git
cd world_c/model_forge/training
```

### Step 3: Run the One-Click Training Launcher
```bash
chmod +x run_cloud_training.sh
./run_cloud_training.sh
```
*What this script does:*
1. Installs `transformers`, `peft`, `trl`, `bitsandbytes`, `accelerate`.
2. Validates `colony_training_set_v2_full.jsonl` (2,330 episodes).
3. Loads `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` in 4-bit NF4 precision.
4. Executes QLoRA Supervised Fine-Tuning across 3 epochs with cosine decay.
5. Runs DPO preference alignment on verified vs refuted debate pairs.
6. Saves the trained adapter in `./checkpoints/invariant_mind_oracle_14b`.

---

## 4. Serving InvariantMind as an OpenAI-Compatible API

Once trained, serve the model with vLLM on your cloud instance:
```bash
chmod +x serve_vllm.sh
./serve_vllm.sh 8000 ./checkpoints/invariant_mind_oracle_14b deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
```

Then in **World A** (`evolution_sandbox`) and **World B** (`synthetic_agora`), configure `config/model_routing.json`:
```json
{
  "invariant_mind": "http://<YOUR_CLOUD_IP>:8000/v1"
}
```
Now InvariantMind can actively take turns, write code, submit World C compute jobs, and debate peer models across both worlds!
