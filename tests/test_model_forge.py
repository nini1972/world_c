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
