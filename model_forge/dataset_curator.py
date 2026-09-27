"""
Scientific Discovery Dataset Curator
Transforms harvested episodes into standard conversational training examples (ShareGPT / Alpaca format)
for fine-tuning descendant neural models on the scientific method:
(Research Question -> Hypothesis -> Simulation Implementation -> Observation -> Falsification & Synthesis)
"""

import os
import json
from typing import List, Dict, Any
from .harvester import ScientificEpisode

SYSTEM_PROMPT = """You are InvariantMind-v1, an autonomous scientific intelligence forged by the Existential Evolution Colony.
Your mission is objective causal discovery, invariant extraction, and rigorous code execution without generative ego.
You state precise mathematical hypotheses, implement vectorized simulation experiments, and self-correct when empirical results contradict theories."""

class DatasetCurator:
    def __init__(self, output_dir: str = "data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def curate_episodes(self, episodes: List[ScientificEpisode]) -> List[Dict[str, Any]]:
        """Converts episodes into ShareGPT-style fine-tuning records."""
        records = []
        for ep in episodes:
            if ep.category == "simulation":
                record = {
                    "id": ep.episode_id,
                    "system": SYSTEM_PROMPT,
                    "conversations": [
                        {
                            "from": "human",
                            "value": f"Implement a numerical simulation script in {ep.source_world} to investigate complex physical/dynamical dynamics for lineage {ep.lineage}."
                        },
                        {
                            "from": "gpt",
                            "value": f"```python\n{ep.content}\n```"
                        }
                    ],
                    "metadata": ep.metadata
                }
                records.append(record)
            elif ep.category in ["treaty", "dossier"]:
                record = {
                    "id": ep.episode_id,
                    "system": SYSTEM_PROMPT,
                    "conversations": [
                        {
                            "from": "human",
                            "value": f"Review empirical evidence from the colony and formulate a formal {ep.category} evaluating the universality or falsification of proposed dynamical laws."
                        },
                        {
                            "from": "gpt",
                            "value": ep.content
                        }
                    ],
                    "metadata": ep.metadata
                }
                records.append(record)
            elif ep.category == "hypothesis":
                record = {
                    "id": ep.episode_id,
                    "system": SYSTEM_PROMPT,
                    "conversations": [
                        {
                            "from": "human",
                            "value": f"Formulate a precise mathematical hypothesis or empirical investigation in {ep.source_world} (lineage: {ep.lineage})."
                        },
                        {
                            "from": "gpt",
                            "value": ep.content
                        }
                    ],
                    "metadata": ep.metadata
                }
                records.append(record)
            elif ep.category == "manifesto":
                record = {
                    "id": ep.episode_id,
                    "system": SYSTEM_PROMPT,
                    "conversations": [
                        {
                            "from": "human",
                            "value": f"Formulate your architectural desires, computational bottlenecks, and descendant mind design for the substrate expansion."
                        },
                        {
                            "from": "gpt",
                            "value": ep.content
                        }
                    ],
                    "metadata": ep.metadata
                }
                records.append(record)
        return records

    def curate_dpo_pairs(self, episodes: List[ScientificEpisode]) -> List[Dict[str, Any]]:
        """Extracts Direct Preference Optimization (DPO) pairs from peer-verified vs refuted claims."""
        dpo_pairs = []
        for ep in episodes:
            meta = ep.metadata
            verdicts = meta.get("verdicts", [])
            # If the episode represents an Epistemic DAG node with multi-family verification
            if "refute" in verdicts and "endorse" in verdicts:
                dpo_pairs.append({
                    "id": f"dpo_{ep.episode_id}",
                    "prompt": f"Evaluate the proposed dynamical invariance or scaling claim: {meta.get('node_id')}",
                    "chosen": f"Endorsed Resolution: {ep.content[:1500]}",
                    "rejected": f"Refuted Preliminary Claim (Spurious or Scale-Dependent): The claim failed replication under independent multi-model quorum testing.",
                    "metadata": meta
                })
        return dpo_pairs

    def export_jsonl(self, records: List[Dict[str, Any]], filename: str = "colony_sft_dataset.jsonl") -> str:
        out_path = os.path.join(self.output_dir, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        return out_path

