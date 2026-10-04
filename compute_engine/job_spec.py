"""
Declarative Job Specification and Lifecycle Models for Compute Engine
"""

from enum import Enum
from dataclasses import dataclass, field, asdict
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
