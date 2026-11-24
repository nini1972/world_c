import os
import json
import pytest

from model_forge.harvester import ColonyDataHarvester, ScientificEpisode
from model_forge.dataset_curator import DatasetCurator
from model_forge.architectures.invariant_mind import InvariantMindConfig

def test_model_forge_dataset_curation(tmp_path):
    episodes = [
        ScientificEpisode(
            episode_id="ep_test_001",
            source_world="world_a",
            lineage="cartographer",
            category="simulation",
            file_path="kuramoto_test.py",
            content="import numpy as np\n# Kuramoto simulation code",
            metadata={"test": True}
        ),
        ScientificEpisode(
            episode_id="ep_test_002",
            source_world="world_b",
            lineage="agora_assembly",
            category="treaty",
            file_path="TREATY_001.md",
            content="# Treaty on Invariant Non-Universality\nRatified.",
            metadata={"treaty_num": 1}
        )
    ]
    
    curator = DatasetCurator(output_dir=str(tmp_path))
    records = curator.curate_episodes(episodes)
    assert len(records) == 2
    assert "conversations" in records[0]
    assert records[0]["conversations"][0]["from"] == "human"
    assert records[0]["conversations"][1]["from"] == "gpt"
    
    out_file = curator.export_jsonl(records, filename="test_dataset.jsonl")
    assert os.path.exists(out_file)
    with open(out_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) == 2

def test_invariant_mind_config():
    config = InvariantMindConfig()
    assert config.model_name == "InvariantMind-v1"
    assert len(config.rg_scales) == 5
    assert "scaling_collapse_contrastive_loss" in config.training_objectives


def test_tool_trajectory_curator_builds_clean_multiturn_hermes_dataset(tmp_path):
    import re
    from model_forge.tool_trajectory_curator import ToolTrajectoryCurator

    wa_root = tmp_path / "world_a"
    wb_root = tmp_path / "world_b"
    out_dir = tmp_path / "data"

    # Good lineage in World A (with one echo/pip_install noise turn interspersed)
    good_log = wa_root / "instances" / "claude_sonnet_4_5" / "logs" / "history.jsonl"
    good_log.parent.mkdir(parents=True)
    # Self-excluded lineage (invariant_mind) must never be read even if passed explicitly
    im_log = wa_root / "instances" / "invariant_mind" / "logs" / "history.jsonl"
    im_log.parent.mkdir(parents=True)

    rows = [
        {"role": "assistant", "content": "Reading channel.", "tool_calls": [{"id": "c1", "function": {"name": "read_chat", "arguments": json.dumps({"channel": "#world-c-coordination"})}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "read_chat", "content": "msg_1 from @xiaomi_mimo"},
        {"role": "assistant", "content": "Install pkg", "tool_calls": [{"id": "bad1", "function": {"name": "run_command", "arguments": json.dumps({"command": "pip install scipy"})}}]},
        {"role": "tool", "tool_call_id": "bad1", "name": "run_command", "content": "ok"},
        {"role": "assistant", "content": "Echo md", "tool_calls": [{"id": "bad2", "function": {"name": "run_command", "arguments": json.dumps({"command": "echo '# Atlas Heading\\n## Section'"})}}]},
        {"role": "tool", "tool_call_id": "bad2", "name": "run_command", "content": "printed"},
        {"role": "assistant", "content": "Writing simulation.", "tool_calls": [{"id": "c2", "function": {"name": "write_file", "arguments": json.dumps({"path": "sweep.py", "content": "print(42)"})}}]},
        {"role": "tool", "tool_call_id": "c2", "name": "write_file", "content": "Successfully wrote sweep.py"},
        {"role": "assistant", "content": "Running World C sweep.", "tool_calls": [{"id": "c3", "function": {"name": "submit_world_c_job", "arguments": json.dumps({"title": "Sweep", "script_content": "print(42)"})}}]},
        {"role": "tool", "tool_call_id": "c3", "name": "submit_world_c_job", "content": "Job submitted: job_123"},
    ]
    good_log.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    im_log.write_text("\n".join(json.dumps(r) for r in rows[:4]) + "\n", encoding="utf-8")

    curator = ToolTrajectoryCurator(
        world_a_root=str(wa_root),
        world_b_root=str(wb_root),
        output_dir=str(out_dir),
    )
    records, stats = curator.curate(
        world_a_lineages=["claude_sonnet_4_5", "invariant_mind"],
        world_b_lineages=[],
    )
    assert len(records) == 1
    assert stats["multi_turn_pct"] == 100.0
    assert stats["hermes_tool_call_pct"] == 100.0
    msgs = records[0]["messages"]
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"

    asst_msgs = [m for m in msgs if m["role"] == "assistant"]
    tool_msgs = [m for m in msgs if m["role"] == "tool"]
    assert len(asst_msgs) == 3
    assert len(tool_msgs) == 3
    payload_text = json.dumps(asst_msgs + tool_msgs)
    assert "pip install" not in payload_text
    assert "Atlas Heading" not in payload_text
    for am in asst_msgs:
        m_tc = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", am["content"], re.DOTALL)
        assert m_tc is not None
        parsed = json.loads(m_tc.group(1))
        assert "name" in parsed and isinstance(parsed["arguments"], dict)
        assert len(am["tool_calls"]) == 1

    out_path = curator.export_jsonl(records, filename="v3_test.jsonl")
    assert out_path.is_file()


def test_train_sft_trims_trailing_tool_turns_and_upload_renders_v3_card(tmp_path):
    from model_forge.training.train_sft import parse_colony_records
    from model_forge.training.upload_to_hf import render_model_card

    ds_file = tmp_path / "sample_v3.jsonl"
    rec = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "call1\n\n<tool_call>\n{\"name\":\"read_chat\",\"arguments\":{\"channel\":\"#lounge\"}}\n</tool_call>"},
            {"role": "tool", "name": "read_chat", "content": "hi"},
            {"role": "assistant", "content": "call2\n\n<tool_call>\n{\"name\":\"post_chat\",\"arguments\":{\"channel\":\"#lounge\",\"message\":\"done\"}}\n</tool_call>"},
            {"role": "tool", "name": "post_chat", "content": "posted"},
        ]
    }
    ds_file.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    parsed = parse_colony_records(str(ds_file))
    assert len(parsed) == 1
    msgs = parsed[0]["messages"]
    assert msgs[-1]["role"] == "assistant"
    assert msgs[3]["role"] == "user"
    assert msgs[3]["content"].startswith("<tool_response>")

    card = render_model_card("worker_v3_14b", "Ninitje/InvariantMind-Worker-v3-14B-Merged")
    assert "Qwen/Qwen2.5-Coder-14B-Instruct" in card
    assert "Ninitje/InvariantMind-Worker-v3-14B-Merged" in card
