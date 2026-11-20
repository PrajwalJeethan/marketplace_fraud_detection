import os

from flask import Flask

from .config import (
    ANALYSIS_JOB_TTL_SECONDS,
    CANDIDATE_LABELS,
    CLASSIFIER_BATCH_SIZE,
    FEEDBACK_VALUES,
    FRAUD_LABELS,
    FRAUD_THRESHOLD,
    HUMAN_REVIEW_LOWER_THRESHOLD,
    HUMAN_REVIEW_UPPER_THRESHOLD,
    MAX_ANALYSIS_JOBS,
    MAX_EXPLANATION_SEGMENTS,
    MAX_TEXT_LENGTH,
    MODEL_ID,
    REVIEW_DECISIONS,
)
from .routes.api import api


def create_app(registry=None):
    app = Flask(__name__)
    app.config["REVIEW_DB_PATH"] = os.getenv(
        "REVIEW_DB_PATH",
        os.path.join(app.instance_path, "review_decisions.sqlite3"),
    )
    app.extensions["marketplace"] = registry or {}
    app.register_blueprint(api)
    return app
