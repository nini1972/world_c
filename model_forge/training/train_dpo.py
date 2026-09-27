"""
Direct Preference Optimization (DPO) Script for InvariantMind-v1
Aligns the model to reject scale-dependent artifacts and endorse quorum-verified invariants
using the Colony's peer-review debate records.
"""

import os
import argparse
import yaml
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from trl import DPOTrainer

def parse_args():
    parser = argparse.ArgumentParser(description="DPO alignment for InvariantMind")
    parser.add_argument("--adapter_path", required=True, help="Path to SFT adapter checkpoint")
    parser.add_argument("--base_model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
    parser.add_argument("--dpo_data", default="../../data/colony_dpo_pairs_v1.jsonl")
    parser.add_argument("--output_dir", default="./checkpoints/invariant_mind_dpo")
    return parser.parse_args()

def load_dpo_dataset(path: str):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    return Dataset.from_list(pairs)

def main():
    args = parse_args()
    print("=" * 60)
    print("🎯 Launching InvariantMind-v1 Direct Preference Optimization (DPO)")
    print(f"Base Model: {args.base_model}")
    print(f"SFT Adapter: {args.adapter_path}")
    print("=" * 60)
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    # Load base model + SFT adapter
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_path, is_trainable=True)
    
    dataset = load_dpo_dataset(args.dpo_data)
    print(f"Loaded {len(dataset)} preference pairs.")
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        num_train_epochs=2,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=5,
        bf16=torch.cuda.is_bf16_supported(),
        save_strategy="epoch",
        gradient_checkpointing=True
    )
    
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None, # PEFT handles reference model automatically
        args=training_args,
        beta=0.1,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_length=2048,
        max_prompt_length=1024
    )
    
    dpo_trainer.train()
    dpo_trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"✅ DPO alignment complete! Saved to {args.output_dir}")

if __name__ == "__main__":
    main()
