"""Backward-compatible Flask entry point.

The application implementation lives in ``src/marketplace_fraud_detection``.
This module keeps the existing ``python app.py`` workflow and test imports.
"""

import os
import sys
import time
from pathlib import Path
from uuid import uuid4


SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from marketplace_fraud_detection import create_app
from marketplace_fraud_detection import config
from marketplace_fraud_detection.repositories import review_repository
from marketplace_fraud_detection.services import job_manager, model_service


MODEL_ID = config.MODEL_ID
FRAUD_LABELS = config.FRAUD_LABELS
CANDIDATE_LABELS = config.CANDIDATE_LABELS
FRAUD_THRESHOLD = config.FRAUD_THRESHOLD
HUMAN_REVIEW_LOWER_THRESHOLD = config.HUMAN_REVIEW_LOWER_THRESHOLD
HUMAN_REVIEW_UPPER_THRESHOLD = config.HUMAN_REVIEW_UPPER_THRESHOLD
MAX_TEXT_LENGTH = config.MAX_TEXT_LENGTH
MAX_ANALYSIS_JOBS = config.MAX_ANALYSIS_JOBS
ANALYSIS_JOB_TTL_SECONDS = config.ANALYSIS_JOB_TTL_SECONDS
MAX_EXPLANATION_SEGMENTS = config.MAX_EXPLANATION_SEGMENTS
CLASSIFIER_BATCH_SIZE = config.CLASSIFIER_BATCH_SIZE
REVIEW_DECISIONS = config.REVIEW_DECISIONS
FEEDBACK_VALUES = config.FEEDBACK_VALUES

REVIEW_DB_PATH = os.getenv(
    "REVIEW_DB_PATH",
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "instance",
        "review_decisions.sqlite3",
    ),
)

analysis_executor = job_manager.analysis_executor
analysis_jobs = job_manager.analysis_jobs
analysis_jobs_lock = job_manager.analysis_jobs_lock


def prune_analysis_jobs():
    job_manager.prune_analysis_jobs()


def get_analysis_job(job_id):
    return job_manager.get_analysis_job(job_id)


def create_analysis_job(function, text):
    job_manager.prune_analysis_jobs()
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


def get_classifier():
    return model_service.get_classifier()


def get_fraud_score(result):
    return model_service.get_fraud_score(result)


def get_fraud_category_scores(result):
    return model_service.get_fraud_category_scores(result)


def get_decision(fraud_score):
    return model_service.get_decision(fraud_score)


def split_explanation_segments(text):
    return model_service.split_explanation_segments(text)


def analyze_listing(text, classifier=None):
    return model_service.analyze_listing(
        text,
        classifier=classifier or get_classifier(),
    )


def explain_with_classifier(
    text, original_score, original_category_scores, classifier=None
):
    return model_service.explain_with_classifier(
        text,
        original_score,
        original_category_scores,
        classifier=classifier or get_classifier(),
    )


def connect_review_db():
    return review_repository.connect_review_db(REVIEW_DB_PATH)


def create_evaluation_tables(connection):
    review_repository.create_evaluation_tables(connection)


def record_human_review(job_id, analysis, decision, reviewer_id, rationale):
    return review_repository.record_human_review(
        REVIEW_DB_PATH,
        job_id,
        analysis,
        decision,
        reviewer_id,
        rationale,
    )


def record_usability_response(payload):
    return review_repository.record_usability_response(REVIEW_DB_PATH, payload)


def record_usefulness_feedback(job_id, feedback):
    return _record_usefulness_feedback(job_id, feedback)


def _record_usefulness_feedback(job_id, feedback):
    feedback_id = uuid4().hex
    connection = connect_review_db()
    try:
        create_evaluation_tables(connection)
        with connection:
            connection.execute(
                """
                INSERT INTO usefulness_feedback (
                    feedback_id, analysis_job_id, feedback_value
                ) VALUES (?, ?, ?)
                """,
                (feedback_id, job_id, feedback),
            )
    finally:
        connection.close()
    return feedback_id


class DynamicExecutor:
    def submit(self, *args, **kwargs):
        return analysis_executor.submit(*args, **kwargs)


registry = {
    "create_analysis_job": lambda *args, **kwargs: create_analysis_job(*args, **kwargs),
    "get_analysis_job": lambda *args, **kwargs: get_analysis_job(*args, **kwargs),
    "analyze_listing": lambda *args, **kwargs: analyze_listing(*args, **kwargs),
    "explain_with_classifier": lambda *args, **kwargs: explain_with_classifier(*args, **kwargs),
    "analysis_executor": DynamicExecutor(),
    "record_human_review": lambda *args, **kwargs: record_human_review(*args, **kwargs),
    "record_usability_response": lambda *args, **kwargs: record_usability_response(*args, **kwargs),
    "record_usefulness_feedback": lambda *args, **kwargs: record_usefulness_feedback(*args, **kwargs),
    "connect_review_db": lambda *args, **kwargs: connect_review_db(*args, **kwargs),
    "create_evaluation_tables": lambda *args, **kwargs: create_evaluation_tables(*args, **kwargs),
}

app = create_app(registry)


if __name__ == "__main__":
    app.run(debug=True)
