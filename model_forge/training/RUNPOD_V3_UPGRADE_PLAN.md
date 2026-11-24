# InvariantMind v3 Tool-Calling & RunPod Upgrade Plan

## 1. Live RunPod Serverless Endpoint Snapshot (`2026-11-24`, read-only audit)

* **Endpoint Name:** `invariant-mind-worker`
* **Endpoint ID:** `nxrwj2zma759vc`
* **Current `MODEL_NAME`:** `Ninitje/InvariantMind-Worker-7B-Merged`
* **GPU Pools:** `AMPERE_24`, `ADA_24`, `AMPERE_80`, `ADA_80_PRO` (`count: 1`, `FLASHBOOT`, `idleTimeout: 720`, `workers: min=0, max=1`)
* **Key Runtime Env Vars:**
  * `MAX_MODEL_LEN=16384`
  * `ENABLE_AUTO_TOOL_CHOICE=true` (already enabled)
  * `TOOL_CALL_PARSER=hermes` (already enabled)
  * `ENABLE_PREFIX_CACHING=false` (recommended to enable: saves cold prompt prefill across 20-turn cycles)

---

## 2. Why v3 Was Needed

`data/colony_training_set_v2_full.jsonl` (`2,330` rows) contained `0%` multi-turn trajectories and `0%` tool calls (`82.6%` standalone Python files, `17.4%` standalone Markdown essays), causing `InvariantMind-Worker-7B-Merged` to lose multi-turn Hermes tool calling and drift into writing Markdown summaries.

`data/colony_training_set_v3_toolcall.jsonl` (curated by `model_forge/tool_trajectory_curator.py`) harvests `2–4`-step real multi-turn tool trajectories (`read_chat`, `post_chat`, `submit_world_c_job`, `run_command`, `write_file` `.py`, `peer_verify_node`, `post_epistemic_node`) formatted natively for Qwen2.5 + Hermes `<tool_call>` / `<tool_response>` parsing.

---

## 3. Step-by-Step v3 Training & Cutover Commands

### Step A — Train v3 QLoRA Adapter on a Cloud GPU Pod
* **7B Worker (fits 24 GB GPU, e.g., RTX 4090 / A5000):**
  ```bash
  cd model_forge/training
  python train_sft.py --tier worker_v3_7b --config config.yaml
  ```
* **14B Upgrade Option (fits 48–80 GB GPU, e.g., A100 80 GB — strongly recommended for frontier competition):**
  ```bash
  cd model_forge/training
  python train_sft.py --tier worker_v3_14b --config config.yaml
  ```

### Step B — Merge Adapter & Upload to Hugging Face
```bash
python upload_to_hf.py \
  --base_model Qwen/Qwen2.5-Coder-7B-Instruct \
  --adapter_dir ./checkpoints/invariant_mind_worker_v3_7b \
  --repo_id Ninitje/InvariantMind-Worker-v3-7B-Merged \
  --merge
```
*(Swap `7B` → `14B` if training `worker_v3_14b`.)*

### Step C — Update RunPod Serverless Endpoint (`nxrwj2zma759vc`)
1. Set `MODEL_NAME=Ninitje/InvariantMind-Worker-v3-7B-Merged` (or `14B`).
2. Keep `ENABLE_AUTO_TOOL_CHOICE=true` and `TOOL_CALL_PARSER=hermes`.
3. Enable `ENABLE_PREFIX_CACHING=true` to reuse shared system-prompt KV cache across turns.
4. *(If switching to 14B in BF16)* remove 24 GB pools (`AMPERE_24`, `ADA_24`) or keep `AMPERE_80`/`ADA_80_PRO` so 14B (~28 GB weights) never hits VRAM OOM.

### Step D — Enable Native Hermes Tool Calling in World A & World B
Once `nxrwj2zma759vc` is serving the v3 merged checkpoint, set:
* Workflow env var `RUNPOD_NATIVE_TOOL_CALLS=true` in `evolution_sandbox` and `synthetic-agora`.
* Update `INVARIANT_MIND` model string in World A (`evolution_sandbox/llm_client.py`) and World B (`synthetic-agora/llm_client.py`) to point to the v3 Hugging Face slug (and update `get_model_context_limits` / `timeout` if using 14B).
