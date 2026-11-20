import csv
import io

from flask import Blueprint, Response, jsonify, render_template, request

from flask import current_app

from .. import config
from ..repositories import review_repository
from ..services import job_manager, model_service


api = Blueprint("api", __name__)


def marketplace_service(name):
    registry = current_app.extensions.get("marketplace", {})
    if name in registry:
        return registry[name]
    defaults = {
        "create_analysis_job": job_manager.create_analysis_job,
        "get_analysis_job": job_manager.get_analysis_job,
        "analyze_listing": model_service.analyze_listing,
        "explain_with_classifier": model_service.explain_with_classifier,
        "analysis_executor": job_manager.analysis_executor,
        "record_human_review": lambda *args: review_repository.record_human_review(
            current_app.config["REVIEW_DB_PATH"], *args
        ),
        "record_usability_response": lambda payload: review_repository.record_usability_response(
            current_app.config["REVIEW_DB_PATH"], payload
        ),
        "record_usefulness_feedback": lambda job_id, feedback: record_usefulness_feedback(
            job_id, feedback
        ),
        "connect_review_db": lambda: review_repository.connect_review_db(
            current_app.config["REVIEW_DB_PATH"]
        ),
        "create_evaluation_tables": review_repository.create_evaluation_tables,
    }
    return defaults[name]


def record_usefulness_feedback(job_id, feedback):
    from uuid import uuid4

    feedback_id = uuid4().hex
    connection = review_repository.connect_review_db(
        current_app.config["REVIEW_DB_PATH"]
    )
    try:
        review_repository.create_evaluation_tables(connection)
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


def csv_response(filename, columns, rows):
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@api.get("/")
def index():
    return render_template(
        "index.html",
        model_id=config.MODEL_ID,
        input_limit=config.MAX_TEXT_LENGTH,
        review_lower_percent=f"{config.HUMAN_REVIEW_LOWER_THRESHOLD:.0%}",
        review_upper_percent=f"{config.HUMAN_REVIEW_UPPER_THRESHOLD:.0%}",
        fraud_threshold_percent=f"{config.FRAUD_THRESHOLD:.0%}",
        job_retention_minutes=config.ANALYSIS_JOB_TTL_SECONDS // 60,
    )


@api.get("/analysis")
def analysis_page():
    return render_template(
        "index.html",
        model_id=config.MODEL_ID,
        input_limit=config.MAX_TEXT_LENGTH,
        review_lower_percent=f"{config.HUMAN_REVIEW_LOWER_THRESHOLD:.0%}",
        review_upper_percent=f"{config.HUMAN_REVIEW_UPPER_THRESHOLD:.0%}",
        fraud_threshold_percent=f"{config.FRAUD_THRESHOLD:.0%}",
        job_retention_minutes=config.ANALYSIS_JOB_TTL_SECONDS // 60,
    )


@api.get("/evaluation")
def evaluation_page():
    return render_template("evaluation.html")


@api.get("/study-summary")
def study_summary_page():
    return render_template("study_summary.html")


@api.get("/transparency")
def transparency_page():
    return render_template(
        "transparency.html",
        model_id=config.MODEL_ID,
        input_limit=config.MAX_TEXT_LENGTH,
        review_lower_percent=f"{config.HUMAN_REVIEW_LOWER_THRESHOLD:.0%}",
        review_upper_percent=f"{config.HUMAN_REVIEW_UPPER_THRESHOLD:.0%}",
        fraud_threshold_percent=f"{config.FRAUD_THRESHOLD:.0%}",
        job_retention_minutes=config.ANALYSIS_JOB_TTL_SECONDS // 60,
    )


@api.post("/api/analyze")
def api_analyze():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="Send a JSON object with a 'text' field."), 400

    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        return jsonify(error="'text' must be a non-empty string."), 400
    if len(text) > config.MAX_TEXT_LENGTH:
        return jsonify(
            error=f"'text' must be at most {config.MAX_TEXT_LENGTH} characters."
        ), 400

    job_id = marketplace_service("create_analysis_job")(
        marketplace_service("analyze_listing"),
        text,
    )
    if job_id is None:
        return jsonify(error="The analysis queue is full. Try again shortly."), 503
    return jsonify(job_id=job_id, status="processing"), 202


@api.get("/api/analyze/<job_id>")
def api_analysis_status(job_id):
    job = marketplace_service("get_analysis_job")(job_id)
    if job is None:
        return jsonify(error="Analysis job was not found or has expired."), 404

    future = job["analysis_future"]
    if not future.done():
        return jsonify(status="processing"), 202
    try:
        return jsonify(future.result())
    except Exception:
        current_app.logger.exception("Marketplace listing classification failed")
        return jsonify(error="The classification model is unavailable."), 503


@api.post("/api/analyze/<job_id>/explain")
def api_start_explanation(job_id):
    job = marketplace_service("get_analysis_job")(job_id)
    if job is None:
        return jsonify(error="Analysis job was not found or has expired."), 404
    if not job["analysis_future"].done():
        return jsonify(error="Wait for the classification result first."), 409
    if job["explanation_future"] is None:
        analysis = job["analysis_future"].result()
        job["explanation_future"] = marketplace_service("analysis_executor").submit(
            marketplace_service("explain_with_classifier"),
            job["text"],
            analysis["fraud_score"],
            analysis["fraud_category_scores"],
        )
    return jsonify(status="processing"), 202


@api.get("/api/analyze/<job_id>/explain")
def api_explanation_status(job_id):
    job = marketplace_service("get_analysis_job")(job_id)
    if job is None:
        return jsonify(error="Analysis job was not found or has expired."), 404
    future = job["explanation_future"]
    if future is None:
        return jsonify(error="Explanation has not been requested."), 409
    if not future.done():
        return jsonify(status="processing"), 202
    try:
        return jsonify(explanation=future.result())
    except Exception:
        current_app.logger.exception("Model explanation failed")
        return jsonify(error="The model explanation is unavailable."), 503


@api.post("/api/analyze/<job_id>/review")
def api_record_review(job_id):
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="Send a JSON object with a review decision."), 400

    decision = payload.get("decision")
    if not isinstance(decision, str) or decision not in config.REVIEW_DECISIONS:
        return jsonify(error="Choose a valid human review decision."), 400

    reviewer_id = payload.get("reviewer_id", "")
    rationale = payload.get("rationale", "")
    if not isinstance(reviewer_id, str) or len(reviewer_id) > 80:
        return jsonify(error="Reviewer ID must be at most 80 characters."), 400
    if not isinstance(rationale, str) or len(rationale) > 500:
        return jsonify(error="Rationale must be at most 500 characters."), 400

    job = marketplace_service("get_analysis_job")(job_id)
    if job is None:
        return jsonify(error="Analysis job was not found or has expired."), 404
    if not job["analysis_future"].done():
        return jsonify(error="Wait for the classification result first."), 409
    try:
        analysis = job["analysis_future"].result()
        review_id = marketplace_service("record_human_review")(
            job_id,
            analysis,
            decision,
            reviewer_id.strip() or None,
            rationale.strip() or None,
        )
    except Exception:
        current_app.logger.exception("Human review could not be recorded")
        return jsonify(error="The review could not be saved."), 503

    return jsonify(
        status="recorded",
        review_id=review_id,
        decision=decision,
        model_label=analysis["label"],
        fraud_score=analysis["fraud_score"],
    ), 201


@api.post("/api/analyze/<job_id>/feedback")
def api_record_usefulness_feedback(job_id):
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="Send a JSON object with a feedback value."), 400
    feedback = payload.get("feedback")
    if feedback not in config.FEEDBACK_VALUES:
        return jsonify(error="Choose 'helpful' or 'not-helpful'."), 400
    job = marketplace_service("get_analysis_job")(job_id)
    if job is None:
        return jsonify(error="Analysis job was not found or has expired."), 404

    feedback_id = marketplace_service("record_usefulness_feedback")(job_id, feedback)
    return jsonify(status="recorded", feedback_id=feedback_id, feedback=feedback), 201


@api.post("/api/evaluation")
def api_record_evaluation():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="Send a JSON object with usability ratings."), 400
    try:
        response_id = marketplace_service("record_usability_response")(payload)
    except ValueError as error:
        return jsonify(error=str(error)), 400
    except Exception:
        current_app.logger.exception("Usability response could not be recorded")
        return jsonify(error="The usability response could not be saved."), 503
    return jsonify(status="recorded", response_id=response_id), 201


@api.get("/api/evaluation/summary")
def api_evaluation_summary():
    connection = marketplace_service("connect_review_db")()
    try:
        marketplace_service("create_evaluation_tables")(connection)
        row = connection.execute(
            """
            SELECT
                COUNT(*) AS response_count,
                AVG(understands_output) AS understands_output,
                AVG(feels_in_control) AS feels_in_control,
                AVG(trusts_appropriately) AS trusts_appropriately,
                AVG(feedback_easy) AS feedback_easy
            FROM usability_responses
            WHERE consent_acknowledged = 1
            """
        ).fetchone()
        feedback = connection.execute(
            """
            SELECT feedback_value, COUNT(*) AS count
            FROM usefulness_feedback
            GROUP BY feedback_value
            """
        ).fetchall()
    finally:
        connection.close()

    feedback_counts = {item["feedback_value"]: item["count"] for item in feedback}
    rating_fields = (
        "understands_output",
        "feels_in_control",
        "trusts_appropriately",
        "feedback_easy",
    )
    averages = {
        field: round(row[field], 2) if row[field] is not None else None
        for field in rating_fields
    }
    return jsonify(
        response_count=row["response_count"],
        averages=averages,
        usefulness_feedback={
            "helpful": feedback_counts.get("helpful", 0),
            "not-helpful": feedback_counts.get("not-helpful", 0),
        },
    )


@api.get("/api/evaluation/export.csv")
def api_export_evaluation():
    connection = marketplace_service("connect_review_db")()
    try:
        marketplace_service("create_evaluation_tables")(connection)
        records = connection.execute(
            """
            SELECT response_id, participant_id, scenario_id, understands_output,
                   feels_in_control, trusts_appropriately, feedback_easy,
                   comments, consent_acknowledged, created_at
            FROM usability_responses
            ORDER BY created_at
            """
        ).fetchall()
    finally:
        connection.close()

    columns = [
        "response_id", "participant_id", "scenario_id", "understands_output",
        "feels_in_control", "trusts_appropriately", "feedback_easy",
        "comments", "consent_acknowledged", "created_at",
    ]
    return csv_response(
        "hcai-usability-responses.csv",
        columns,
        [dict(record) for record in records],
    )


@api.get("/api/reviews/export.csv")
def api_export_reviews():
    connection = marketplace_service("connect_review_db")()
    try:
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
        records = connection.execute(
            """
            SELECT review_id, analysis_job_id, model_id, model_label,
                   fraud_score, review_decision, reviewer_id, rationale, created_at
            FROM human_reviews
            ORDER BY created_at
            """
        ).fetchall()
    finally:
        connection.close()

    columns = [
        "review_id", "analysis_job_id", "model_id", "model_label",
        "fraud_score", "review_decision", "reviewer_id", "rationale",
        "created_at",
    ]
    return csv_response(
        "hcai-human-review-decisions.csv",
        columns,
        [dict(record) for record in records],
    )
