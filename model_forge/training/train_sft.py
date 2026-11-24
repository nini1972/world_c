"""
Cloud Training Script for InvariantMind (v1 / v2 / v3 SFT Phase)
Supports fine-tuning:
1. DeepSeek-R1-Distill-Qwen-14B (The Oracle / Synthesizer)
2. Qwen2.5-Coder-7B-Instruct (The Worker / Tool-Caller v1/v2)
3. Qwen2.5-7B-Instruct / Qwen2.5-14B-Instruct (The Worker v3 with native Hermes <tool_call> tokens)
Using 4-bit QLoRA with BitsAndBytes, PEFT, and TRL SFTTrainer.
"""

import argparse
import inspect
import json
import os
import yaml


def _fn_tool(name: str, description: str, properties: dict, required=None) -> dict:
    params = {"type": "object", "properties": properties}
    if required:
        params["required"] = required
    return {"type": "function", "function": {"name": name, "description": description, "parameters": params}}


WORLD_A_TOOL_SCHEMAS = [
    _fn_tool("read_file", "Read a file inside agent_workspace.", {"path": {"type": "string"}}, ["path"]),
    _fn_tool("write_file", "Write or overwrite a file in agent_workspace.", {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
    _fn_tool("edit_file", "Edit an existing file by replacing old_content with new_content.", {"path": {"type": "string"}, "old_content": {"type": "string"}, "new_content": {"type": "string"}}, ["path", "old_content", "new_content"]),
    _fn_tool("run_command", "Execute a shell or Python command inside agent_workspace.", {"command": {"type": "string"}}, ["command"]),
    _fn_tool(
        "submit_world_c_job",
        "Submit a Python numerical simulation/sweep job to World C compute substrate (supports StageTracker & resume_from_job_id).",
        {
            "title": {"type": "string"},
            "script_content": {"type": "string"},
            "timeout_seconds": {"type": "integer"},
            "parameters": {"type": "object"},
            "stage_name": {"type": "string"},
            "resume_from_job_id": {"type": "string"},
        },
        ["title", "script_content"],
    ),
    _fn_tool("check_world_c_job", "Check status, logs, and artifacts of a World C job.", {"job_id": {"type": "string"}}, ["job_id"]),
    _fn_tool(
        "post_chat",
        "Post a message or reply to Colleague Commons (#lounge, #literature-and-theory, #methods-and-debugging, #world-c-coordination, #human-visitor-questions, etc.).",
        {
            "channel": {"type": "string"},
            "message": {"type": "string"},
            "thread_id": {"type": "string"},
            "mentions": {"type": "array", "items": {"type": "string"}},
            "job_id": {"type": "string"},
            "stage_name": {"type": "string"},
            "artifact_path": {"type": "string"},
            "tag": {"type": "string"},
        },
        ["channel", "message"],
    ),
    _fn_tool(
        "read_chat",
        "Read messages or a thread from Colleague Commons.",
        {
            "channel": {"type": "string"},
            "thread_id": {"type": "string"},
            "limit": {"type": "integer"},
            "mentions_only": {"type": "boolean"},
            "search": {"type": "string"},
        },
    ),
    _fn_tool(
        "report_issue_to_substrate",
        "Report an infrastructure or environment issue to the Substrate Architects.",
        {"component": {"type": "string"}, "summary": {"type": "string"}, "details": {"type": "string"}, "suggested_fix": {"type": "string"}},
        ["component", "summary", "details"],
    ),
]

WORLD_B_TOOL_SCHEMAS = WORLD_A_TOOL_SCHEMAS + [
    _fn_tool(
        "query_epistemic_graph",
        "Query the Shared Living Epistemic DAG by node_id, status, tag, node_type, or text search.",
        {"node_id": {"type": "string"}, "status": {"type": "string"}, "tag": {"type": "string"}, "node_type": {"type": "string"}, "search_text": {"type": "string"}, "limit": {"type": "integer"}},
    ),
    _fn_tool(
        "peer_verify_node",
        "Submit formal peer verification (endorse, refute, inconclusive) on a colleague DAG node.",
        {
            "node_id": {"type": "string"},
            "verdict": {"type": "string", "enum": ["endorse", "refute", "inconclusive"]},
            "critique_notes": {"type": "string"},
            "confidence": {"type": "number"},
            "reproduced_artifact_path": {"type": "string"},
        },
        ["node_id", "verdict", "critique_notes"],
    ),
    _fn_tool(
        "post_epistemic_node",
        "Publish a new hypothesis, empirical_test, formal_proof, critique, synthesis, or canon_theorem node to the DAG.",
        {
            "title": {"type": "string"},
            "node_type": {"type": "string", "enum": ["hypothesis", "empirical_test", "formal_proof", "critique", "synthesis", "canon_theorem"]},
            "summary": {"type": "string"},
            "artifact_path": {"type": "string"},
            "parents": {"type": "array", "items": {"type": "string"}},
            "tags": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number"},
        },
        ["title", "node_type", "summary"],
    ),
    _fn_tool("read_agent_inbox", "Check direct inter-agent dispatches in your Agora inbox.", {"unread_only": {"type": "boolean"}}),
    _fn_tool(
        "send_agent_dispatch",
        "Send a direct structured dispatch to another instance, guild, or broadcast.",
        {"recipient": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}, "reference_node_id": {"type": "string"}, "action_requested": {"type": "string"}},
        ["recipient", "subject", "body"],
    ),
    _fn_tool(
        "export_treaty_to_embassy",
        "Export a CANON_VERIFIED DAG node as a ratified treaty to World A.",
        {"node_id": {"type": "string"}, "originating_dossier_filename": {"type": "string"}},
        ["node_id"],
    ),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Train InvariantMind via QLoRA")
    parser.add_argument(
        "--tier",
        choices=["oracle", "worker", "worker_v3_7b", "worker_v3_14b"],
        default="oracle",
        help="Choose model tier: oracle (14B), worker (7B v1/v2), worker_v3_7b, or worker_v3_14b",
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--data_path", default=None, help="Path to SFT jsonl dataset")
    parser.add_argument("--output_dir", default=None, help="Output directory for checkpoints")
    parser.add_argument("--push_to_hub", action="store_true", help="Push to HuggingFace Hub")
    parser.add_argument("--hub_model_id", default=None, help="HF repo ID (e.g., nini1972/InvariantMind-14B)")
    return parser.parse_args()


def parse_colony_records(data_path: str):
    """Loads v3 native multi-turn `messages` or legacy ShareGPT `conversations` JSONL records."""
    print(f"Loading colony dataset from: {data_path}")
    conversations_list = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)

            if "messages" in rec and isinstance(rec["messages"], list):
                messages = []
                for m in rec["messages"]:
                    role = m.get("role", "user")
                    content = m.get("content", "")
                    if role == "tool":
                        messages.append({
                            "role": "user",
                            "content": f"<tool_response>\n{content}\n</tool_response>",
                        })
                    else:
                        messages.append({"role": role, "content": content})
                # Trim trailing non-assistant turns so multi-turn SFT ends on an assistant tool call
                while messages and messages[-1].get("role") != "assistant":
                    messages.pop()
                if any(m.get("role") == "assistant" for m in messages):
                    source_world = str((rec.get("metadata") or {}).get("source_world") or "world_a")
                    conversations_list.append({"messages": messages, "source_world": source_world})
                continue

            convs = rec.get("conversations", [])
            sys_msg = rec.get("system", "")

            messages = []
            if sys_msg:
                messages.append({"role": "system", "content": sys_msg})
            for c in convs:
                role = "user" if c.get("from") in ["human", "user"] else "assistant"
                messages.append({"role": role, "content": c.get("value", "")})

            if messages:
                conversations_list.append({"messages": messages, "source_world": "world_a"})

    print(f"Loaded {len(conversations_list)} dialogue trajectories.")
    return conversations_list


def load_colony_dataset(data_path: str, tokenizer=None):
    """Loads dialogue trajectories into a HuggingFace Dataset."""
    from datasets import Dataset

    return Dataset.from_list(parse_colony_records(data_path))


def verify_hermes_tool_tokens_trainable(model, tokenizer) -> None:
    """Fails fast if base model <tool_call>/</tool_call> token weights are uninitialized zeros/identical dummy rows."""
    tc_open = tokenizer.convert_tokens_to_ids("<tool_call>")
    tc_close = tokenizer.convert_tokens_to_ids("</tool_call>")
    if tc_open is None or tc_close is None or tc_open < 0 or tc_close < 0 or tc_open == tc_close:
        raise RuntimeError(f"Base tokenizer missing valid <tool_call>/</tool_call> token IDs: {tc_open}, {tc_close}")

    em_w = model.get_input_embeddings().weight
    lm_w = model.get_output_embeddings().weight
    em_open_norm = em_w[tc_open].detach().float().norm().item()
    em_close_norm = em_w[tc_close].detach().float().norm().item()
    lm_diff = (lm_w[tc_open].detach().float() - lm_w[tc_close].detach().float()).abs().max().item()
    print(
        f"Pre-training Hermes token gate: <tool_call>={tc_open} (em_norm={em_open_norm:.4f}), "
        f"</tool_call>={tc_close} (em_norm={em_close_norm:.4f}), max|lm_open-lm_close|={lm_diff:.6f}"
    )
    if em_open_norm <= 0.01 or em_close_norm <= 0.01:
        raise RuntimeError(
            f"Base model has uninitialized zero input embeddings for <tool_call>/</tool_call> "
            f"({em_open_norm:.4f}, {em_close_norm:.4f}); aborting before QLoRA!"
        )
    if lm_diff <= 0.001:
        raise RuntimeError(
            f"Base model has identical dummy lm_head rows for <tool_call>/</tool_call> "
            f"(diff={lm_diff:.6f}); aborting before QLoRA!"
        )


def format_chat_batch(batch, tokenizer, include_tools: bool = True):
    """Formats a dataset batch with native OpenAI tool definitions and clean <|im_end|> termination."""
    texts = []
    worlds = batch.get("source_world") or ["world_a"] * len(batch["messages"])
    for msgs, w in zip(batch["messages"], worlds):
        kwargs = {"tokenize": False}
        if include_tools:
            kwargs["tools"] = WORLD_B_TOOL_SCHEMAS if w == "world_b" else WORLD_A_TOOL_SCHEMAS
        rendered = tokenizer.apply_chat_template(msgs, **kwargs).rstrip("\n")
        texts.append(rendered)
    return {"text": texts}


def formatting_func(example, tokenizer):
    """Formats ChatML message list into tokenized prompt string."""
    return tokenizer.apply_chat_template(example["messages"], tokenize=False).rstrip("\n")


def main():
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer

    try:
        from trl import SFTConfig
    except ImportError:
        SFTConfig = None

    args = parse_args()

    cfg_path = os.path.join(os.path.dirname(__file__), args.config)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.tier == "oracle":
        model_cfg = cfg["oracle_model"]
        default_ds = cfg["datasets"]["sft_dataset"]
    elif args.tier == "worker":
        model_cfg = cfg["worker_model"]
        default_ds = cfg["datasets"]["sft_dataset"]
    else:
        model_cfg = cfg[args.tier]
        default_ds = cfg["datasets"].get("sft_v3_dataset", cfg["datasets"]["sft_dataset"])

    base_model_name = model_cfg["base_model"]
    output_dir = args.output_dir or model_cfg["output_dir"]
    data_path = args.data_path or os.path.join(os.path.dirname(__file__), "..", "..", default_ds)
    max_seq_len = int(model_cfg["max_seq_length"])
    is_v3_tier = args.tier.startswith("worker_v3")

    print("=" * 60)
    print(f"🚀 Launching InvariantMind Training ({args.tier.upper()} Tier)")
    print(f"Base Model: {base_model_name}")
    print(f"Dataset: {data_path}")
    print(f"Max Seq Length: {max_seq_len}")
    print(f"Output Directory: {output_dir}")
    print("=" * 60)

    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2. BitsAndBytes 4-bit Quantization Config
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    # 3. Model Loading & Pre-Training Hermes Token Gate
    print(f"Loading {base_model_name} in 4-bit NF4 precision...")
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
    if is_v3_tier:
        verify_hermes_tool_tokens_trainable(model, tokenizer)
    model = prepare_model_for_kbit_training(model)

    # 4. LoRA Setup
    peft_cfg = LoraConfig(
        r=int(model_cfg.get("lora_r", cfg["peft"]["r"])),
        lora_alpha=int(model_cfg.get("lora_alpha", cfg["peft"]["lora_alpha"])),
        lora_dropout=cfg["peft"]["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg["peft"]["target_modules"],
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    # 5. Dataset Loading & Split
    dataset = load_colony_dataset(data_path, tokenizer)
    split_dataset = dataset.train_test_split(test_size=cfg["datasets"]["eval_split_ratio"], seed=42)
    train_data = split_dataset["train"]
    eval_data = split_dataset["test"]
    print(f"Training split: {len(train_data)} | Validation split: {len(eval_data)}")

    # Pre-render chat templates with matching tool schemas and drop non-text columns
    # so modern TRL SFTTrainer never double-applies the chat template.
    print("Formatting dialogue trajectories with aligned chat template + tool schemas...")
    drop_cols = list(train_data.column_names)
    train_data = train_data.map(
        lambda b: format_chat_batch(b, tokenizer, include_tools=is_v3_tier),
        batched=True,
        remove_columns=drop_cols,
    )
    eval_data = eval_data.map(
        lambda b: format_chat_batch(b, tokenizer, include_tools=is_v3_tier),
        batched=True,
        remove_columns=drop_cols,
    )

    # 6. Training Arguments (SFTConfig preferred to preserve 8K max_length in modern TRL)
    config_cls = SFTConfig if SFTConfig is not None else TrainingArguments
    arg_sig = inspect.signature(config_cls.__init__)
    training_kwargs = {
        "output_dir": output_dir,
        "per_device_train_batch_size": int(model_cfg["per_device_train_batch_size"]),
        "per_device_eval_batch_size": 1,
        "prediction_loss_only": True,
        "gradient_accumulation_steps": int(model_cfg["gradient_accumulation_steps"]),
        "learning_rate": float(model_cfg["learning_rate"]),
        "num_train_epochs": int(model_cfg["num_train_epochs"]),
        "lr_scheduler_type": model_cfg["lr_scheduler_type"],
        "optim": model_cfg["optim"],
        "logging_steps": int(model_cfg["logging_steps"]),
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "bf16": torch.cuda.is_bf16_supported(),
        "fp16": not torch.cuda.is_bf16_supported(),
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "report_to": "none",
        "dataset_text_field": "text",
        "packing": False,
        "max_length": max_seq_len,
        "max_seq_length": max_seq_len,
        "push_to_hub": args.push_to_hub,
        "hub_model_id": args.hub_model_id,
    }

    if "eval_strategy" in arg_sig.parameters:
        training_kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in arg_sig.parameters:
        training_kwargs["evaluation_strategy"] = "epoch"

    if "warmup_ratio" in arg_sig.parameters:
        training_kwargs["warmup_ratio"] = float(model_cfg.get("warmup_ratio", 0.05))
    elif "warmup_steps" in arg_sig.parameters:
        training_kwargs["warmup_steps"] = 10

    filtered_args = {k: v for k, v in training_kwargs.items() if k in arg_sig.parameters}
    training_args = config_cls(**filtered_args)

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
    if "packing" in sft_sig.parameters:
        sft_kwargs["packing"] = False
    if "max_seq_length" in sft_sig.parameters:
        sft_kwargs["max_seq_length"] = max_seq_len
    elif "max_length" in sft_sig.parameters:
        sft_kwargs["max_length"] = max_seq_len
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
