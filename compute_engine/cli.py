"""
Command-Line Interface for World C Compute Engine
Usage:
    python -m compute_engine.cli list
    python -m compute_engine.cli status <job_id>
    python -m compute_engine.cli submit <path_to_spec.json>
"""

import sys
import json
import argparse
from .job_spec import JobSpec
from .dispatcher import ComputeDispatcher

def main():
    parser = argparse.ArgumentParser(description="World C Compute Engine CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # List
    list_p = subparsers.add_parser("list", help="List all jobs")
    
    # Status
    status_p = subparsers.add_parser("status", help="Check status of a job")
    status_p.add_argument("job_id", help="ID of the job")
    
    # Submit
    submit_p = subparsers.add_parser("submit", help="Submit a job specification")
    submit_p.add_argument("spec_file", help="Path to job JSON spec")
    submit_p.add_argument("--run", action="store_true", help="Execute immediately")
    
    args = parser.parse_args()
    dispatcher = ComputeDispatcher()
    
    if args.command == "list":
        jobs = dispatcher.list_jobs()
        print(f"Total jobs: {len(jobs)}")
        for j in jobs:
            print(f"[{j.get('status')}] {j.get('job_id')} - Exec time: {j.get('execution_time_seconds', 0):.2f}s")
    elif args.command == "status":
        res = dispatcher.get_result(args.job_id)
        if not res:
            print(f"Job {args.job_id} not found.")
        else:
            print(json.dumps(res.to_dict(), indent=2))
    elif args.command == "submit":
        with open(args.spec_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        spec = JobSpec(**data)
        jid = dispatcher.submit(spec)
        print(f"Job submitted successfully with ID: {jid}")
        if args.run:
            print(f"Executing job {jid}...")
            res = dispatcher.execute_sync(jid)
            print(f"Job finished with status: {res.status.value}")

if __name__ == "__main__":
    main()
