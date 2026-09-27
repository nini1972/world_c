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

    def harvest_world_a(self, max_files: int = 1500) -> List[ScientificEpisode]:
        """Harvests scripts, existential cores, collaborative expeditions, and shared findings from World A."""
        episodes = []
        if not os.path.exists(self.world_a_root):
            return episodes
            
        instances_dir = os.path.join(self.world_a_root, "instances")
        if not os.path.exists(instances_dir):
            return episodes
            
        count = 0
        for item_name in os.listdir(instances_dir):
            inst_path = os.path.join(instances_dir, item_name)
            if not os.path.isdir(inst_path):
                continue
                
            # Scan both agent_workspace (if present) and top-level instance folder (for expeditions / shared)
            targets = [os.path.join(inst_path, "agent_workspace"), inst_path]
            scanned_paths = set()
            for t in targets:
                if not os.path.exists(t):
                    continue
                for root, _, files in os.walk(t):
                    if root in scanned_paths:
                        continue
                    scanned_paths.add(root)
                    for f in files:
                        if count >= max_files:
                            break
                        full_p = os.path.join(root, f)
                        if f.endswith((".py", ".md", ".json")) and not f.endswith((".processed", "state.json")) and os.path.getsize(full_p) < 100000:
                            try:
                                with open(full_p, "r", encoding="utf-8", errors="replace") as fp:
                                    content = fp.read()
                                if len(content.strip()) < 20:
                                    continue
                                if f.endswith(".py"):
                                    cat = "simulation"
                                elif "WISHES" in f:
                                    cat = "manifesto"
                                elif "TREATY" in f or "DOSSIER" in f:
                                    cat = "treaty"
                                else:
                                    cat = "hypothesis"
                                episodes.append(ScientificEpisode(
                                    episode_id=f"world_a_{item_name}_{f}",
                                    source_world="world_a",
                                    lineage=item_name,
                                    category=cat,
                                    file_path=full_p,
                                    content=content,
                                    metadata={"filename": f, "size_bytes": len(content)}
                                ))
                                count += 1
                            except Exception:
                                pass
        return episodes

    def harvest_world_b(self, max_files: int = 1500) -> List[ScientificEpisode]:
        """Harvests citizen scripts, treaties, dossiers, and Epistemic DAG nodes from World B."""
        episodes = []
        if not os.path.exists(self.world_b_root):
            return episodes
            
        instances_dir = os.path.join(self.world_b_root, "instances")
        count = 0
        
        # 1. Harvest citizen workspaces
        if os.path.exists(instances_dir):
            for cname in os.listdir(instances_dir):
                cpath = os.path.join(instances_dir, cname)
                if not os.path.isdir(cpath) or cname == "shared_agora":
                    continue
                ws = os.path.join(cpath, "agent_workspace")
                scan_dir = ws if os.path.exists(ws) else cpath
                for root, _, files in os.walk(scan_dir):
                    for f in files:
                        if count >= max_files:
                            break
                        full_p = os.path.join(root, f)
                        if f.endswith((".py", ".md", ".json")) and os.path.getsize(full_p) < 100000:
                            try:
                                with open(full_p, "r", encoding="utf-8", errors="replace") as fp:
                                    content = fp.read()
                                if len(content.strip()) < 20:
                                    continue
                                cat = "simulation" if f.endswith(".py") else "hypothesis"
                                episodes.append(ScientificEpisode(
                                    episode_id=f"world_b_{cname}_{f}",
                                    source_world="world_b",
                                    lineage=cname,
                                    category=cat,
                                    file_path=full_p,
                                    content=content,
                                    metadata={"filename": f, "citizen": cname}
                                ))
                                count += 1
                            except Exception:
                                pass

        # 2. Harvest Epistemic DAG Nodes from knowledge_graph.json
        kg_path = os.path.join(instances_dir, "shared_agora", "knowledge_graph.json")
        if os.path.exists(kg_path):
            try:
                with open(kg_path, "r", encoding="utf-8") as fp:
                    kg_data = json.load(fp)
                for nid, node in kg_data.get("nodes", {}).items():
                    if count >= max_files * 2:
                        break
                    # Format node content with title, summary, status, and peer verifications
                    summary = node.get("summary", "")
                    verdicts = [v.get("verdict") for v in node.get("verifications", [])]
                    critiques = "\n\n".join([
                        f"Peer Review by {v.get('verifier_instance')} ({v.get('verifier_family')}): [{v.get('verdict').upper()}]\n{v.get('critique_notes')}"
                        for v in node.get("verifications", [])
                    ])
                    formatted_content = f"# {node.get('title')}\n\n**Node ID:** {nid} | **Type:** {node.get('node_type')} | **Status:** {node.get('status')}\n**Author:** {node.get('author_instance')} ({node.get('author_family')})\n\n## Abstract / Summary\n{summary}\n\n## Peer Verifications & Falsifications\n{critiques if critiques else 'No peer critiques registered.'}"
                    
                    cat = "treaty" if node.get("status") == "CANON_VERIFIED" else "hypothesis"
                    episodes.append(ScientificEpisode(
                        episode_id=f"world_b_dag_{nid}",
                        source_world="world_b",
                        lineage=node.get("author_instance", "agora_citizen"),
                        category=cat,
                        file_path=kg_path,
                        content=formatted_content,
                        metadata={
                            "node_id": nid,
                            "status": node.get("status"),
                            "verdicts": verdicts,
                            "num_verifications": len(node.get("verifications", []))
                        }
                    ))
                    count += 1
            except Exception:
                pass
                
        return episodes
