"""
Quick Inference Test Script for InvariantMind-v1 (Post-DPO)
Tests the fine-tuned adapter on a complex colony synthesis question.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

def main():
    base_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    adapter_path = "./checkpoints/invariant_mind_dpo"
    
    print("====================================================================")
    print("🧠 Loading InvariantMind-v1 (DPO-Aligned Oracle Reasoner)...")
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
    
    print(f"Attaching DPO adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    
    prompt = (
        "You are InvariantMind, an autonomous digital scientist forged from 25,000 collective research turns across World A and World B. "
        "Analyze the Kuramoto synchronization phase transition and the phi^4 relativistic kink collisions. "
        "What are the true mathematical invariances, and how do we distinguish genuine physical laws from finite-size computational artifacts?"
    )
    
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
    
    print("\n" + "=" * 60)
    print("🔮 InvariantMind-v1 Chain-of-Thought & Synthesis:")
    print("=" * 60 + "\n")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=600,
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
