"""
Declarative Job Specification and Lifecycle Models for Compute Engine
"""

from enum import Enum
from dataclasses import dataclass, field, asdict, fields
from typing import Dict, Any, List, Optional
import time
import uuid

class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

@dataclass
class JobSpec:
    job_id: str = field(default_factory=lambda: f"job_{uuid.uuid4().hex[:8]}")
    title: str = "Untitled Scientific Job"
    lineage_author: str = "anonymous"
    realm_source: str = "world_a"
    entrypoint: str = "main.py"
    script_content: str = ""
    arguments: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 3600
    checkpoint_interval_seconds: int = 60
    env_vars: Dict[str, str] = field(default_factory=dict)
    output_artifacts: List[str] = field(default_factory=list)
    stage_name: Optional[str] = None
    resume_from_job_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)

    def __post_init__(self):
        try:
            self.timeout_seconds = int(self.timeout_seconds)
        except (ValueError, TypeError):
            self.timeout_seconds = 3600
        try:
            self.checkpoint_interval_seconds = int(self.checkpoint_interval_seconds)
        except (ValueError, TypeError):
            self.checkpoint_interval_seconds = 60
        if not isinstance(self.parameters, dict):
            self.parameters = {}
        # Allow stage_name and resume_from_job_id to be passed either at top-level or inside parameters
        if not self.stage_name:
            for k in ("stage_name", "stage", "stage_id", "project_stage"):
                if self.parameters.get(k):
                    self.stage_name = str(self.parameters[k]).strip()
                    break
        if not self.resume_from_job_id:
            for k in ("resume_from_job_id", "resume_from", "parent_job_id", "previous_job_id"):
                if self.parameters.get(k):
                    self.resume_from_job_id = str(self.parameters[k]).strip()
                    break

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobSpec":
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        # Preserve any extra top-level keys inside parameters so nothing is lost
        extra = {k: v for k, v in data.items() if k not in valid_keys}
        if extra:
            params = dict(filtered.get("parameters") or {})
            for ek, ev in extra.items():
                params.setdefault(ek, ev)
            filtered["parameters"] = params
        return cls(**filtered)

@dataclass
class JobResult:
    job_id: str
    status: JobStatus = JobStatus.QUEUED
    exit_code: Optional[int] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    execution_time_seconds: float = 0.0
    stdout_tail: str = ""
    stderr_tail: str = ""
    error_message: Optional[str] = None
    artifacts_generated: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d
