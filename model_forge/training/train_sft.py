"""
Cloud Training Script for InvariantMind-v1 (SFT Phase)
Supports fine-tuning:
1. DeepSeek-R1-Distill-Qwen-14B (The Oracle / Synthesizer)
2. Qwen2.5-Coder-7B-Instruct (The Worker / Tool-Caller)
Using 4-bit QLoRA with BitsAndBytes, PEFT, and TRL SFTTrainer.
"""

import os
import sys
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
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train InvariantMind via QLoRA")
    parser.add_argument("--tier", choices=["oracle", "worker"], default="oracle",
                        help="Choose model tier: oracle (14B) or worker (7B)")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--data_path", default=None, help="Path to SFT jsonl dataset")
    parser.add_argument("--output_dir", default=None, help="Output directory for checkpoints")
    parser.add_argument("--push_to_hub", action="store_true", help="Push to HuggingFace Hub")
    parser.add_argument("--hub_model_id", default=None, help="HF repo ID (e.g., nini1972/InvariantMind-14B)")
    return parser.parse_args()

def load_colony_dataset(data_path: str, tokenizer):
    """Loads ShareGPT-style jsonl dataset and applies tokenizer chat template."""
    print(f"Loading colony dataset from: {data_path}")
    conversations_list = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            convs = rec.get("conversations", [])
            sys_msg = rec.get("system", "")
            
            # Convert to standard ChatML messages
            messages = []
            if sys_msg:
                messages.append({"role": "system", "content": sys_msg})
            for c in convs:
                role = "user" if c.get("from") in ["human", "user"] else "assistant"
                messages.append({"role": role, "content": c.get("value", "")})
                
            conversations_list.append({"messages": messages})
            
    print(f"Loaded {len(conversations_list)} dialogue trajectories.")
    dataset = Dataset.from_list(conversations_list)
    return dataset

def formatting_func(example, tokenizer):
    """Formats ChatML message list into tokenized prompt string."""
    return tokenizer.apply_chat_template(example["messages"], tokenize=False)

def main():
    args = parse_args()
    
    # Load config
    cfg_path = os.path.join(os.path.dirname(__file__), args.config)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    model_cfg = cfg["oracle_model"] if args.tier == "oracle" else cfg["worker_model"]
    base_model_name = model_cfg["base_model"]
    output_dir = args.output_dir or model_cfg["output_dir"]
    data_path = args.data_path or os.path.join(os.path.dirname(__file__), "..", "..", cfg["datasets"]["sft_dataset"])
    
    print("=" * 60)
    print(f"🚀 Launching InvariantMind-v1 Training ({args.tier.upper()} Tier)")
    print(f"Base Model: {base_model_name}")
    print(f"Output Directory: {output_dir}")
    print("=" * 60)
    
    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 2. BitsAndBytes 4-bit Quantization Config
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    # 3. Model Loading
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Loading {base_model_name} in 4-bit NF4 precision...")
    import inspect
    model_init_sig = inspect.signature(AutoModelForCausalLM.from_pretrained)
    model_kwargs = {
        "quantization_config": bnb_cfg,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if "dtype" in model_init_sig.parameters:
        model_kwargs["dtype"] = compute_dtype
    else:
        model_kwargs["torch_dtype"] = compute_dtype

    model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)
    model = prepare_model_for_kbit_training(model)
    
    # 4. LoRA Setup
    peft_cfg = LoraConfig(
        r=cfg["peft"]["r"],
        lora_alpha=cfg["peft"]["lora_alpha"],
        lora_dropout=cfg["peft"]["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg["peft"]["target_modules"]
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()
    
    # 5. Dataset Loading & Split
    dataset = load_colony_dataset(data_path, tokenizer)
    split_dataset = dataset.train_test_split(test_size=cfg["datasets"]["eval_split_ratio"], seed=42)
    train_data = split_dataset["train"]
    eval_data = split_dataset["test"]
    print(f"Training split: {len(train_data)} | Validation split: {len(eval_data)}")

    # Pre-render chat templates into a robust 'text' column
    print("Formatting dialogue trajectories with chat template...")
    def format_chat(batch):
        return {"text": [tokenizer.apply_chat_template(m, tokenize=False) for m in batch["messages"]]}
    
    train_data = train_data.map(format_chat, batched=True)
    eval_data = eval_data.map(format_chat, batched=True)
    
    # 6. Training Arguments (Dynamically filtered for version tolerance)
    arg_sig = inspect.signature(TrainingArguments.__init__)
    training_kwargs = {
        "output_dir": output_dir,
        "per_device_train_batch_size": model_cfg["per_device_train_batch_size"],
        "gradient_accumulation_steps": model_cfg["gradient_accumulation_steps"],
        "learning_rate": float(model_cfg["learning_rate"]),
        "num_train_epochs": model_cfg["num_train_epochs"],
        "lr_scheduler_type": model_cfg["lr_scheduler_type"],
        "optim": model_cfg["optim"],
        "logging_steps": model_cfg["logging_steps"],
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "bf16": torch.cuda.is_bf16_supported(),
        "fp16": not torch.cuda.is_bf16_supported(),
        "gradient_checkpointing": True,
        "report_to": ["tensorboard"],
        "push_to_hub": args.push_to_hub,
        "hub_model_id": args.hub_model_id
    }
    
    if "eval_strategy" in arg_sig.parameters:
        training_kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in arg_sig.parameters:
        training_kwargs["evaluation_strategy"] = "epoch"
        
    if "warmup_ratio" in arg_sig.parameters:
        training_kwargs["warmup_ratio"] = float(model_cfg.get("warmup_ratio", 0.05))
    elif "warmup_steps" in arg_sig.parameters:
        training_kwargs["warmup_steps"] = float(model_cfg.get("warmup_ratio", 0.05))
        
    filtered_args = {k: v for k, v in training_kwargs.items() if k in arg_sig.parameters}
    training_args = TrainingArguments(**filtered_args)
    
    # 7. SFT Trainer
    sft_sig = inspect.signature(SFTTrainer.__init__)
    sft_kwargs = {
        "model": model,
        "train_dataset": train_data,
        "eval_dataset": eval_data,
        "args": training_args,
    }
    if "dataset_text_field" in sft_sig.parameters:
        sft_kwargs["dataset_text_field"] = "text"
    if "max_seq_length" in sft_sig.parameters:
        sft_kwargs["max_seq_length"] = model_cfg["max_seq_length"]
    if "tokenizer" in sft_sig.parameters:
        sft_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in sft_sig.parameters:
        sft_kwargs["processing_class"] = tokenizer

    trainer = SFTTrainer(**sft_kwargs)
    
    print("Training initiated...")
    trainer.train()
    
    print(f"Saving final adapter to: {output_dir}")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    if args.push_to_hub and args.hub_model_id:
        print(f"Pushing adapter to Hugging Face Hub: {args.hub_model_id}")
        trainer.model.push_to_hub(args.hub_model_id)
        tokenizer.push_to_hub(args.hub_model_id)
        
    print("✅ Training completed successfully!")

if __name__ == "__main__":
    main()
