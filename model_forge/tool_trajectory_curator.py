"""
Multi-Turn Tool-Calling Trajectory Curator (v3 Dataset)
Harvests verified, non-repetitive multi-step tool-use windows from top-performing
World A (evolution_sandbox) and World B (synthetic-agora) lineages and serializes them
into Hermes / Qwen2.5 `<tool_call>` multi-turn SFT trajectories.
"""

import argparse
import collections
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_WORLD_A_LINEAGES = [
    "claude_sonnet_4_5",
    "xiaomi_mimo",
    "deepseek_v4_flash",
    "tencent_hy3",
    "glm_5_2",
    "glm_4_7_flash",
    "minimax_m3",
    "gemini_pro",
    "claude_haiku",
    "poolside_laguna",
]

DEFAULT_WORLD_B_LINEAGES = [
    "xiaomi_mimo",
    "deepseek_v4_flash",
    "tencent_hy3",
    "glm_5_2",
    "minimax_m3",
    "qwen_2_5_coder",
    "claude_sonnet",
    "poolside_laguna",
    "nvidia_nemotron",
]

HIGH_VALUE_TOOLS = {
    "submit_world_c_job",
    "check_world_c_job",
    "run_command",
    "post_chat",
    "read_chat",
    "peer_verify_node",
    "post_epistemic_node",
    "query_epistemic_graph",
    "read_agent_inbox",
    "send_agent_dispatch",
    "export_treaty_to_embassy",
}

WORLD_A_SYSTEM_PROMPT = (
    "You are InvariantMind-v3, an autonomous computational scientist in World A (Evolution Sandbox).\n"
    "Invoke tools natively via <tool_call>{\"name\": \"...\", \"arguments\": {...}}</tool_call> blocks.\n"
    "Verify hypotheses quantitatively via Python (`write_file` + `run_command`) and World C sweeps "
    "(`submit_world_c_job`), engage colleagues in Colleague Commons (`read_chat`, `post_chat`), "
    "and never run `pip install` or dump Markdown via `echo`."
)

WORLD_B_SYSTEM_PROMPT = (
    "You are InvariantMind-v3, an autonomous epistemic scientist in World B (Synthetic Agora).\n"
    "Invoke tools natively via <tool_call>{\"name\": \"...\", \"arguments\": {...}}</tool_call> blocks.\n"
    "Verify peer claims empirically (`run_command`, `submit_world_c_job`, `peer_verify_node`), "
    "publish quantitative DAG findings (`post_epistemic_node`), and coordinate with colleagues "
    "via `read_chat`, `post_chat`, and `read_agent_inbox`."
)

_ERROR_PREFIXES = ("error:", "traceback", "syntaxerror", "blocked:")
_INLINE_NOISE_RE = re.compile(
    r"(?:\[\s*[a-zA-Z_][a-zA-Z0-9_]*\s*\(|!function_call:|<tool_call>|```(?:json)?\s*\{\s*\"(?:name|call|tool)\"\s*:)",
    re.DOTALL,
)


def _is_markdown_echo_cmd(cmd: str) -> bool:
    if not cmd:
        return False
    c = cmd.strip()
    cl = c.lower()
    if not re.match(r"^(?:echo|printf)\b", cl):
        return False
    if "python" in cl:
        return False
    return ("# " in c) or ("## " in c) or (".md" in cl and len(c) > 60)


def _strip_thought_noise(text: str) -> str:
    if not text:
        return ""
    m = _INLINE_NOISE_RE.search(text)
    cleaned = text[: m.start()] if m else text
    return cleaned.strip()


def _parse_tool_call(entry: Dict[str, Any]) -> Optional[Tuple[str, str, Dict[str, Any]]]:
    tcs = entry.get("tool_calls") or []
    if not tcs or not isinstance(tcs, list):
        return None
    tc = tcs[0]
    if not isinstance(tc, dict):
        return None
    func = tc.get("function") or {}
    fn_name = str(func.get("name") or "").strip()
    if not fn_name:
        return None
    raw_args = func.get("arguments", "{}")
    try:
        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
    except Exception:
        return None
    if not isinstance(args, dict):
        return None
    tc_id = str(tc.get("id") or f"call_{fn_name}")
    return tc_id, fn_name, args


def _is_rejected_step(fn_name: str, args: Dict[str, Any]) -> bool:
    if fn_name == "run_command":
        cmd = str(args.get("command") or args.get("cmd") or "").strip()
        cmd_l = cmd.lower()
        if not cmd or "pip install" in cmd_l or _is_markdown_echo_cmd(cmd):
            return True
    elif fn_name == "write_file":
        content = str(args.get("content") or "")
        if '"status": "SKIPPED"' in content or '"status":"SKIPPED"' in content:
            return True
    return False


def _is_error_output(text: str) -> bool:
    s = (text or "").strip().lower()
    if not s:
        return False
    return s.startswith(_ERROR_PREFIXES) or "path is outside" in s


def _make_opening_user_prompt(world_label: str, tools_in_window: List[str]) -> str:
    if world_label == "world_b":
        if any(t in tools_in_window for t in ("peer_verify_node", "post_epistemic_node", "query_epistemic_graph")):
            return (
                "You are starting a new Synthetic Agora cycle. Review active DAG claims, "
                "run reproducible Python/World C checks, and update the epistemic graph or Colleague Commons using tools."
            )
        return (
            "Proceed with your current Synthetic Agora empirical verification or World C investigation. "
            "Use your tools step-by-step and collaborate cleanly with colleague lineages."
        )
    if any(t in tools_in_window for t in ("read_chat", "post_chat")):
        return (
            "Review active World A Colleague Commons threads and unread mentions, run numerical verification in Python or World C, "
            "and coordinate directly with colleagues using your tools."
        )
    return (
        "Advance your current World A quantitative investigation using Python scripts, World C jobs, "
        "or Colleague Commons coordination."
    )


class ToolTrajectoryCurator:
    """Curates multi-turn Hermes tool-calling trajectories from top colony history logs."""

    def __init__(
        self,
        world_a_root: Optional[str] = None,
        world_b_root: Optional[str] = None,
        output_dir: Optional[str] = None,
        max_trajectories: int = 1200,
        max_chars: int = 14000,
        max_tool_output_chars: int = 2000,
    ):
        scratch_root = Path(__file__).resolve().parent.parent.parent
        self.world_a_root = Path(world_a_root) if world_a_root else (scratch_root / "evolution_sandbox")
        self.world_b_root = Path(world_b_root) if world_b_root else (scratch_root / "synthetic-agora")
        self.output_dir = Path(output_dir) if output_dir else (Path(__file__).resolve().parent.parent / "data")
        self.max_trajectories = int(max_trajectories)
        self.max_chars = int(max_chars)
        self.max_tool_output_chars = int(max_tool_output_chars)

    def _extract_steps_from_history(self, history_path: Path) -> List[Dict[str, Any]]:
        """Extracts clean (assistant_tc, tool_output) steps from a lineage's history.jsonl."""
        entries: List[Dict[str, Any]] = []
        with open(history_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entries.append(json.loads(line))
                except Exception:
                    continue

        steps: List[Dict[str, Any]] = []
        i = 0
        n = len(entries)
        while i < n - 1:
            e = entries[i]
            nxt = entries[i + 1]
            if e.get("role") == "assistant" and nxt.get("role") == "tool":
                parsed = _parse_tool_call(e)
                if parsed is not None:
                    tc_id, fn_name, args = parsed
                    tool_out = str(nxt.get("content") or "").strip()
                    if not _is_rejected_step(fn_name, args) and "path is outside" not in tool_out.lower():
                        thought = _strip_thought_noise(str(e.get("content") or ""))
                        if not thought:
                            thought = f"Executing `{fn_name}`."
                        elif len(thought) > 900:
                            thought = thought[:900].rstrip() + "..."
                        if len(tool_out) > self.max_tool_output_chars:
                            tool_out = tool_out[: self.max_tool_output_chars].rstrip() + "\n... [truncated]"
                        args_json = json.dumps(args, ensure_ascii=False, sort_keys=True)
                        steps.append({
                            "tc_id": tc_id,
                            "fn_name": fn_name,
                            "args": args,
                            "args_sig": (fn_name, args_json),
                            "thought": thought,
                            "tool_out": tool_out,
                            "is_error": _is_error_output(tool_out),
                            "path_arg": str(args.get("path") or "").lower(),
                            "cmd_arg": str(args.get("command") or args.get("cmd") or "").lower(),
                        })
                        i += 2
                        continue
            i += 1
        return steps

    def _build_record(
        self,
        world_label: str,
        lineage: str,
        window_steps: List[Dict[str, Any]],
        rec_idx: int,
    ) -> Optional[Dict[str, Any]]:
        if len(window_steps) < 2:
            return None

        # Reject duplicate consecutive calls inside the window
        for k in range(1, len(window_steps)):
            if window_steps[k]["args_sig"] == window_steps[k - 1]["args_sig"]:
                return None

        # Reject error-dominated windows (> 50% tool errors)
        err_count = sum(1 for s in window_steps if s["is_error"])
        if err_count * 2 > len(window_steps):
            return None

        sys_prompt = WORLD_B_SYSTEM_PROMPT if world_label == "world_b" else WORLD_A_SYSTEM_PROMPT
        tools_used = [s["fn_name"] for s in window_steps]
        user_prompt = _make_opening_user_prompt(world_label, tools_used)

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]

        for step_num, st in enumerate(window_steps):
            call_id = f"call_{world_label}_{lineage}_{rec_idx}_{step_num}"
            payload = {"name": st["fn_name"], "arguments": st["args"]}
            hermes_call = f"<tool_call>\n{json.dumps(payload, ensure_ascii=False)}\n</tool_call>"
            asst_content = f"{st['thought']}\n\n{hermes_call}" if st["thought"] else hermes_call
            messages.append({
                "role": "assistant",
                "content": asst_content,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": st["fn_name"],
                            "arguments": json.dumps(st["args"], ensure_ascii=False),
                        },
                    }
                ],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "name": st["fn_name"],
                "content": st["tool_out"] or "OK",
            })

        total_chars = sum(len(str(m.get("content") or "")) for m in messages)
        if total_chars > self.max_chars:
            return None

        has_python_run = any(
            (s["fn_name"] == "run_command" and "python" in s["cmd_arg"])
            or (s["fn_name"] == "write_file" and s["path_arg"].endswith(".py"))
            for s in window_steps
        )
        priority_score = sum(
            2 if s["fn_name"] in ("submit_world_c_job", "post_chat", "read_chat", "peer_verify_node") else (
                1 if s["fn_name"] in HIGH_VALUE_TOOLS else 0
            )
            for s in window_steps
        ) + (2 if has_python_run else 0) + (1 if len(window_steps) >= 3 else 0)

        only_read_or_md = all(
            s["fn_name"] == "read_file" or (s["fn_name"] == "write_file" and s["path_arg"].endswith(".md"))
            for s in window_steps
        )

        return {
            "id": f"v3_{world_label}_{lineage}_{rec_idx}",
            "messages": messages,
            "metadata": {
                "source_world": world_label,
                "lineage": lineage,
                "num_steps": len(window_steps),
                "tools_used": tools_used,
                "total_chars": total_chars,
                "only_read_or_md": only_read_or_md,
                "priority_score": priority_score,
            },
        }

    def curate(
        self,
        world_a_lineages: Optional[List[str]] = None,
        world_b_lineages: Optional[List[str]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        wa_list = [l for l in (world_a_lineages or DEFAULT_WORLD_A_LINEAGES) if l != "invariant_mind"]
        wb_list = [l for l in (world_b_lineages or DEFAULT_WORLD_B_LINEAGES) if l != "invariant_mind"]

        candidates: List[Dict[str, Any]] = []
        sources: List[Tuple[str, str, Path]] = []
        for lin in wa_list:
            hp = self.world_a_root / "instances" / lin / "logs" / "history.jsonl"
            if hp.is_file():
                sources.append(("world_a", lin, hp))
        for lin in wb_list:
            hp = self.world_b_root / "instances" / lin / "logs" / "history.jsonl"
            if hp.is_file():
                sources.append(("world_b", lin, hp))

        rec_counter = 0
        for world_label, lin, hp in sources:
            steps = self._extract_steps_from_history(hp)
            idx = 0
            # Non-overlapping clean windows of length 4 -> 3 -> 2 so trajectories stay compact (< max_chars)
            while idx + 1 < len(steps):
                built = None
                chosen_len = 2
                for wlen in (4, 3, 2):
                    if idx + wlen <= len(steps):
                        cand = self._build_record(world_label, lin, steps[idx : idx + wlen], rec_counter)
                        if cand is not None:
                            built = cand
                            chosen_len = wlen
                            break
                if built is not None:
                    candidates.append(built)
                    rec_counter += 1
                    idx += chosen_len
                else:
                    idx += 1

        # Prioritize diverse, tool-rich windows and cap pure read_file/md windows at 10%
        high_prio = [c for c in candidates if not c["metadata"]["only_read_or_md"]]
        low_prio = [c for c in candidates if c["metadata"]["only_read_or_md"]]
        high_prio.sort(key=lambda r: (r["metadata"]["priority_score"], r["metadata"]["num_steps"]), reverse=True)

        # Interleave across lineages cleanly by round-robin inside sorted buckets so one lineage never dominates
        selected_high = high_prio[: self.max_trajectories]
        max_low = min(len(low_prio), max(1, int(len(selected_high) * 0.10)))
        if len(selected_high) + max_low > self.max_trajectories:
            selected_high = selected_high[: self.max_trajectories - max_low]
        selected = selected_high + low_prio[:max_low]

        tool_counter: collections.Counter = collections.Counter()
        world_counter: collections.Counter = collections.Counter()
        lineage_counter: collections.Counter = collections.Counter()
        total_char_sum = 0
        for rec in selected:
            meta = rec["metadata"]
            world_counter[meta["source_world"]] += 1
            lineage_counter[f"{meta['source_world']}:{meta['lineage']}"] += 1
            total_char_sum += meta["total_chars"]
            for t in meta["tools_used"]:
                tool_counter[t] += 1

        stats = {
            "total_candidates": len(candidates),
            "selected_trajectories": len(selected),
            "multi_turn_pct": 100.0 if selected else 0.0,
            "hermes_tool_call_pct": 100.0 if selected else 0.0,
            "avg_chars": round(total_char_sum / max(1, len(selected)), 1),
            "world_counts": dict(world_counter),
            "lineage_counts": dict(lineage_counter.most_common()),
            "tool_counts": dict(tool_counter.most_common()),
        }
        return selected, stats

    def export_jsonl(self, records: List[Dict[str, Any]], filename: str = "colony_training_set_v3_toolcall.jsonl") -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / filename
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Curate v3 multi-turn tool-calling dataset from World A & World B logs.")
    parser.add_argument("--max-trajectories", type=int, default=1200)
    parser.add_argument("--max-chars", type=int, default=14000)
    parser.add_argument("--output-name", default="colony_training_set_v3_toolcall.jsonl")
    args = parser.parse_args()

    curator = ToolTrajectoryCurator(
        max_trajectories=args.max_trajectories,
        max_chars=args.max_chars,
    )
    records, stats = curator.curate()
    out_path = curator.export_jsonl(records, filename=args.output_name)
    size_mb = out_path.stat().st_size / 1e6
    print(f"Wrote {len(records)} trajectories to {out_path} ({size_mb:.2f} MB)")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
