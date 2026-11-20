import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from uuid import uuid4

from ..config import ANALYSIS_JOB_TTL_SECONDS, MAX_ANALYSIS_JOBS


analysis_executor = ThreadPoolExecutor(max_workers=1)
analysis_jobs = {}
analysis_jobs_lock = Lock()


def prune_analysis_jobs():
    cutoff = time.monotonic() - ANALYSIS_JOB_TTL_SECONDS
    with analysis_jobs_lock:
        expired_job_ids = [
            job_id
            for job_id, job in analysis_jobs.items()
            if job["analysis_future"].done() and job["created_at"] < cutoff
        ]
        for job_id in expired_job_ids:
            del analysis_jobs[job_id]


def create_analysis_job(function, text):
    prune_analysis_jobs()
    with analysis_jobs_lock:
        if len(analysis_jobs) >= MAX_ANALYSIS_JOBS:
            return None
        job_id = uuid4().hex
        future = analysis_executor.submit(function, text.strip())
        analysis_jobs[job_id] = {
            "analysis_future": future,
            "explanation_future": None,
            "created_at": time.monotonic(),
            "text": text.strip(),
        }
    return job_id


def get_analysis_job(job_id):
    prune_analysis_jobs()
    with analysis_jobs_lock:
        return analysis_jobs.get(job_id)
