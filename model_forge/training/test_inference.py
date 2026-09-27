"""
Quick Inference Test Script for InvariantMind-v1 (Oracle & Worker Tiers)
Tests the fine-tuned adapter on complex scientific reasoning and simulation tasks.
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser(description="Test InvariantMind Inference")
    parser.add_argument("--tier", choices=["oracle", "worker"], default="worker",
                        help="Choose model tier: oracle (14B) or worker (7B)")
    parser.add_argument("--adapter_dir", default=None, help="Path to adapter checkpoint")
    args = parser.parse_args()

    if args.tier == "worker":
        base_model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
        adapter_path = args.adapter_dir or "./checkpoints/invariant_mind_worker_7b"
        prompt = (
            "You are InvariantMind-Worker, the specialized scientific engineer of the autonomous colony. "
            "Write an optimized, vectorized Python implementation using NumPy to simulate N coupled Kuramoto oscillators: "
            "d theta_i / dt = omega_i + (K/N) * sum_j sin(theta_j - theta_i). "
            "Vectorize the phase coupling, compute the complex order parameter R(t) = |(1/N) sum exp(i*theta_j)| at each step, "
            "and verify energy/conservation invariants."
        )
    else:
        base_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
        adapter_path = args.adapter_dir or "./checkpoints/invariant_mind_dpo"
        prompt = (
            "You are InvariantMind, an autonomous digital scientist forged from 25,000 collective research turns across World A and World B. "
            "Analyze the Kuramoto synchronization phase transition and the phi^4 relativistic kink collisions. "
            "What are the true mathematical invariances, and how do we distinguish genuine physical laws from finite-size computational artifacts?"
        )

    print("====================================================================")
    print(f"🧠 Loading InvariantMind-v1 ({args.tier.upper()} Tier: {base_model_name})...")
    print(f"📦 Adapter path: {adapter_path}")
    print("====================================================================")

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_quant_type="nf4"
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True
    )

    print(f"Attaching {args.tier} adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")

    print("\n" + "=" * 60)
    print(f"🔮 InvariantMind-v1 ({args.tier.upper()}) Generation:")
    print("=" * 60 + "\n")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=650,
            temperature=0.6,
            top_p=0.95,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(generated_text)
    print("\n" + "=" * 60)
    print("🎉 Test Generation Complete!")
    print("====================================================================")

if __name__ == "__main__":
    main()

