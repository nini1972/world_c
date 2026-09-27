"""
Compute Engine: The Asynchronous Compute & Parameter Sweep Substrate for World C.
Decouples heavy scientific computation from LLM turn timeouts.
"""

from .job_spec import JobSpec, JobStatus, JobResult
from .checkpoint import CheckpointManager
from .dispatcher import ComputeDispatcher

__all__ = [
    "JobSpec",
    "JobStatus",
    "JobResult",
    "CheckpointManager",
    "ComputeDispatcher",
]
