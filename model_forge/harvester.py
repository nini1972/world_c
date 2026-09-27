"""
Colony Data Harvester
Harvests scientific discovery episodes from World A (evolution_sandbox) and World B (synthetic_agora),
extracting python scripts, hypotheses, empirical results, and treaty falsifications.
"""

import os
import json
import glob
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional

@dataclass
class ScientificEpisode:
    episode_id: str
    source_world: str
    lineage: str
    category: str  # "simulation", "hypothesis", "treaty", "dossier", "manifesto"
    file_path: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class ColonyDataHarvester:
    def __init__(
        self,
        world_a_root: Optional[str] = None,
        world_b_root: Optional[str] = None
    ):
        self.world_a_root = world_a_root or os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "evolution_sandbox")
        )
        self.world_b_root = world_b_root or os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "synthetic_agora")
        )

    def harvest_all(self, max_files_per_category: int = 500) -> List[ScientificEpisode]:
        episodes = []
        episodes.extend(self.harvest_world_a(max_files=max_files_per_category))
        episodes.extend(self.harvest_world_b(max_files=max_files_per_category))
        return episodes

    def harvest_world_a(self, max_files: int = 500) -> List[ScientificEpisode]:
        """Harvests scripts, existential cores, and shared space findings from World A."""
        episodes = []
        if not os.path.exists(self.world_a_root):
            return episodes
            
        instances_dir = os.path.join(self.world_a_root, "instances")
        if not os.path.exists(instances_dir):
            return episodes
            
        count = 0
        for instance_name in os.listdir(instances_dir):
            inst_path = os.path.join(instances_dir, instance_name)
            ws_path = os.path.join(inst_path, "agent_workspace")
            if not os.path.exists(ws_path):
                # Could be shared_space
                ws_path = inst_path
                
            for root, _, files in os.walk(ws_path):
                for f in files:
                    if count >= max_files:
                        break
                    full_p = os.path.join(root, f)
                    if f.endswith((".py", ".md", ".json")) and os.path.getsize(full_p) < 100000:
                        try:
                            with open(full_p, "r", encoding="utf-8", errors="replace") as fp:
                                content = fp.read()
                            cat = "simulation" if f.endswith(".py") else ("manifesto" if "WISHES" in f else "hypothesis")
                            episodes.append(ScientificEpisode(
                                episode_id=f"world_a_{instance_name}_{f}",
                                source_world="world_a",
                                lineage=instance_name,
                                category=cat,
                                file_path=full_p,
                                content=content,
                                metadata={"filename": f, "size_bytes": len(content)}
                            ))
                            count += 1
                        except Exception:
                            pass
        return episodes

    def harvest_world_b(self, max_files: int = 500) -> List[ScientificEpisode]:
        """Harvests treaties, dossiers, and verification scripts from World B."""
        episodes = []
        if not os.path.exists(self.world_b_root):
            return episodes
            
        shared_space = os.path.join(self.world_b_root, "shared_space")
        count = 0
        if os.path.exists(shared_space):
            for root, _, files in os.walk(shared_space):
                for f in files:
                    if count >= max_files:
                        break
                    full_p = os.path.join(root, f)
                    if f.endswith((".md", ".json", ".py")) and os.path.getsize(full_p) < 100000:
                        try:
                            with open(full_p, "r", encoding="utf-8", errors="replace") as fp:
                                content = fp.read()
                            cat = "treaty" if "treaty" in f.lower() or "crt" in f.lower() else ("dossier" if "dossier" in f.lower() else "governance")
                            episodes.append(ScientificEpisode(
                                episode_id=f"world_b_shared_{f}",
                                source_world="world_b",
                                lineage="agora_assembly",
                                category=cat,
                                file_path=full_p,
                                content=content,
                                metadata={"filename": f, "size_bytes": len(content)}
                            ))
                            count += 1
                        except Exception:
                            pass
        return episodes
