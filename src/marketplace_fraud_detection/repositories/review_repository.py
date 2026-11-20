import os
import sqlite3
from uuid import uuid4


def connect_review_db(database_path):
    os.makedirs(os.path.dirname(os.path.abspath(database_path)), exist_ok=True)
    connection = sqlite3.connect(database_path, timeout=5)
    connection.row_factory = sqlite3.Row
    return connection


def create_evaluation_tables(connection):
    with connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS usability_responses (
                response_id TEXT PRIMARY KEY,
                participant_id TEXT,
                scenario_id TEXT,
                understands_output INTEGER NOT NULL,
                feels_in_control INTEGER NOT NULL,
                trusts_appropriately INTEGER NOT NULL,
                feedback_easy INTEGER NOT NULL,
                comments TEXT,
                consent_acknowledged INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS usefulness_feedback (
                feedback_id TEXT PRIMARY KEY,
                analysis_job_id TEXT,
                feedback_value TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def ensure_human_review_table(connection):
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS human_reviews (
            review_id TEXT PRIMARY KEY,
            analysis_job_id TEXT NOT NULL,
            model_id TEXT NOT NULL,
            model_label TEXT NOT NULL,
            fraud_score REAL NOT NULL,
            review_decision TEXT NOT NULL,
            reviewer_id TEXT,
            rationale TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


def record_human_review(database_path, job_id, analysis, decision, reviewer_id, rationale):
    review_id = uuid4().hex
    connection = connect_review_db(database_path)
    try:
        with connection:
            ensure_human_review_table(connection)
            connection.execute(
                """
                INSERT INTO human_reviews (
                    review_id, analysis_job_id, model_id, model_label, fraud_score,
                    review_decision, reviewer_id, rationale
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    review_id,
                    job_id,
                    analysis["model"],
                    analysis["label"],
                    analysis["fraud_score"],
                    decision,
                    reviewer_id,
                    rationale,
                ),
            )
    finally:
        connection.close()
    return review_id


def record_usability_response(database_path, payload):
    required_ratings = (
        "understands_output",
        "feels_in_control",
        "trusts_appropriately",
        "feedback_easy",
    )
    for field in required_ratings:
        value = payload.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
            raise ValueError(f"{field} must be an integer from 1 to 5.")
    if payload.get("consent_acknowledged") is not True:
        raise ValueError("Participant consent must be acknowledged.")

    participant_id = payload.get("participant_id", "")
    scenario_id = payload.get("scenario_id", "")
    comments = payload.get("comments", "")
    if not isinstance(participant_id, str) or len(participant_id) > 80:
        raise ValueError("Participant ID must be at most 80 characters.")
    if not isinstance(scenario_id, str) or len(scenario_id) > 80:
        raise ValueError("Scenario ID must be at most 80 characters.")
    if not isinstance(comments, str) or len(comments) > 1000:
        raise ValueError("Comments must be at most 1000 characters.")

    response_id = uuid4().hex
    connection = connect_review_db(database_path)
    try:
        create_evaluation_tables(connection)
        with connection:
            connection.execute(
                """
                INSERT INTO usability_responses (
                    response_id, participant_id, scenario_id, understands_output,
                    feels_in_control, trusts_appropriately, feedback_easy, comments,
                    consent_acknowledged
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    response_id,
                    participant_id.strip() or None,
                    scenario_id.strip() or None,
                    payload["understands_output"],
                    payload["feels_in_control"],
                    payload["trusts_appropriately"],
                    payload["feedback_easy"],
                    comments.strip() or None,
                ),
            )
    finally:
        connection.close()
    return response_id
