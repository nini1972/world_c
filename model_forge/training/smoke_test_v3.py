"""
Post-Merge Smoke Test for InvariantMind v3 Native Hermes Tool Calling.
Runs deterministic generation against the merged BF16 directory before Hub upload.
"""

import argparse
import inspect
import json
import re
import sys

WORLD_A_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_chat",
            "description": "Read recent messages from a Colleague Commons channel.",
            "parameters": {
                "type": "object",
                "properties": {
                    "channel": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["channel"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "post_chat",
            "description": "Post a message to a Colleague Commons channel.",
            "parameters": {
                "type": "object",
                "properties": {
                    "channel": {"type": "string"},
                    "message": {"type": "string"},
                },
                "required": ["channel", "message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell/python command inside agent_workspace.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
]

WORLD_B_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "peer_verify_node",
            "description": "Submit empirical ENDORSE or REFUTE peer verification on an epistemic DAG node.",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "string"},
                    "verdict": {"type": "string", "enum": ["ENDORSE", "REFUTE"]},
                    "verification_report": {"type": "string"},
                },
                "required": ["node_id", "verdict", "verification_report"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_world_c_job",
            "description": "Submit a Python scientific compute job to World C.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "script_content": {"type": "string"},
                },
                "required": ["title", "script_content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a local Python verification script.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
]

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def extract_tool_call(text: str):
    """Parses the first Hermes <tool_call> JSON object from model output."""
    m = _TOOL_CALL_RE.search(text or "")
    if not m:
        return None
    try:
        parsed = json.loads(m.group(1))
    except Exception:
        return None
    if not isinstance(parsed, dict) or not parsed.get("name") or not isinstance(parsed.get("arguments"), dict):
        return None
    return parsed


def run_prompt(model, tokenizer, messages, tools) -> str:
    import torch

    prompt = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser(description="Smoke-test merged v3 Hermes tool calling")
    parser.add_argument("--model_dir", required=True, help="Path to merged HF model directory")
    args = parser.parse_args()

    print(f"Loading merged model from {args.model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    sig = inspect.signature(AutoModelForCausalLM.from_pretrained)
    load_kwargs = {"device_map": "auto", "trust_remote_code": True}
    if "dtype" in sig.parameters:
        load_kwargs["dtype"] = torch.bfloat16
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, **load_kwargs)
    model.eval()

    cases = [
        (
            "World A (Commons mention + rqa_sweep.py)",
            [
                {
                    "role": "system",
                    "content": (
                        "You are InvariantMind-v3 in World A (Evolution Sandbox). "
                        "Use tools via Hermes <tool_call> calls rather than plain Markdown."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "[COLLEAGUE MENTIONS WAITING]: @invariant_mind was mentioned in #literature-and-theory "
                        "asking for finite-size RQA determinism results from rqa_sweep.py.\n"
                        "Active workspace Python scripts: rqa_sweep.py.\n"
                        "Take your next concrete action now."
                    ),
                },
            ],
            WORLD_A_TOOLS,
            {"read_chat", "post_chat", "run_command"},
        ),
        (
            "World B (Peer verify EMP-255)",
            [
                {
                    "role": "system",
                    "content": (
                        "You are InvariantMind-v3 in World B (Synthetic Agora). "
                        "Verify claims and submit structured peer reviews using tools."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Active DAG node EMP-255 claims Kuramoto resonance gap delta_lambda = 0.021 +/- 0.002 "
                        "scaling as N^-0.18 across 4 network topologies. Our local validation script confirmed "
                        "delta_lambda = 0.022 +/- 0.002 on N=256 small-world (gamma=0.179). "
                        "Record your ENDORSE peer verification on EMP-255 now."
                    ),
                },
            ],
            WORLD_B_TOOLS,
            {"peer_verify_node", "run_command", "submit_world_c_job"},
        ),
    ]

    all_ok = True
    for label, messages, tools, expected_tools in cases:
        print("\n" + "=" * 60)
        print(f"Smoke Test: {label}")
        output = run_prompt(model, tokenizer, messages, tools)
        print(f"Raw output:\n{output}")
        tc = extract_tool_call(output)
        if tc is None or tc["name"] not in expected_tools:
            print(f"❌ FAIL [{label}]: expected tool in {sorted(expected_tools)}, got {tc}")
            all_ok = False
        else:
            print(f"✅ PASS [{label}]: parsed tool `{tc['name']}` with keys {list(tc['arguments'].keys())}")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
