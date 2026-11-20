import os
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from threading import Lock
from uuid import uuid4

from flask import Flask, jsonify, render_template, request


MODEL_ID = os.getenv("HF_MODEL_ID", "typeform/distilbert-base-uncased-mnli")
FRAUD_LABELS = (
    "a potentially fraudulent marketplace listing",
    "advance payment scam",
    "phishing scam",
    "fake payment verification request",
    "fake courier service scam",
)
CANDIDATE_LABELS = [
    *FRAUD_LABELS,
    "legitimate marketplace sale",
    "a legitimate marketplace listing",
]
FRAUD_THRESHOLD = 0.7
HUMAN_REVIEW_LOWER_THRESHOLD = 0.3
HUMAN_REVIEW_UPPER_THRESHOLD = FRAUD_THRESHOLD
MAX_TEXT_LENGTH = 5000
MAX_ANALYSIS_JOBS = 20
ANALYSIS_JOB_TTL_SECONDS = 900
MAX_EXPLANATION_SEGMENTS = 3
CLASSIFIER_BATCH_SIZE = 8
REVIEW_DECISIONS = {
    "confirmed_suspicious",
    "marked_legitimate",
    "escalated",
}

app = Flask(__name__)
REVIEW_DB_PATH = os.getenv(
    "REVIEW_DB_PATH",
    os.path.join(app.instance_path, "review_decisions.sqlite3"),
)
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


def record_human_review(job_id, analysis, decision, reviewer_id, rationale):
    os.makedirs(os.path.dirname(os.path.abspath(REVIEW_DB_PATH)), exist_ok=True)
    review_id = uuid4().hex
    connection = sqlite3.connect(REVIEW_DB_PATH, timeout=5)
    try:
        with connection:
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


def get_fraud_score(result):
    scores = dict(zip(result["labels"], result["scores"]))
    return max(scores.get(label, 0.0) for label in FRAUD_LABELS)


def get_fraud_category_scores(result):
    scores = dict(zip(result["labels"], result["scores"]))
    return {label: float(scores.get(label, 0.0)) for label in FRAUD_LABELS}


def get_decision(fraud_score):
    review_required = (
        HUMAN_REVIEW_LOWER_THRESHOLD
        < fraud_score
        < HUMAN_REVIEW_UPPER_THRESHOLD
    )
    flagged = fraud_score >= FRAUD_THRESHOLD
    if review_required:
        label = "human_review_required"
    elif flagged:
        label = "potentially_fraudulent"
    else:
        label = "likely_legitimate"
    return label, review_required, flagged


def split_explanation_segments(text):
    segments = [
        segment.strip()
        for segment in re.split(r"(?<=[.!?])\s+|[\r\n]+|(?<=[,;])\s*", text.strip())
        if segment.strip()
    ]
    if len(segments) <= 1:
        words = text.split()
        chunk_size = max(1, (len(words) + 2) // 3)
        segments = [
            " ".join(words[index:index + chunk_size])
            for index in range(0, len(words), chunk_size)
        ]
    if len(segments) > MAX_EXPLANATION_SEGMENTS:
        group_size = (len(segments) + MAX_EXPLANATION_SEGMENTS - 1) // MAX_EXPLANATION_SEGMENTS
        segments = [
            " ".join(segments[index:index + group_size])
            for index in range(0, len(segments), group_size)
        ]
    return segments


def explain_with_classifier(text, original_score, original_category_scores, classifier=None):
    classifier = classifier or get_classifier()
    segments = split_explanation_segments(text)
    masked_texts = [
        " ".join(
            segment
            for segment_index, segment in enumerate(segments)
            if segment_index != index
        )
        for index in range(len(segments))
    ]
    masked_texts = [masked_text for masked_text in masked_texts if masked_text]
    if not masked_texts:
        return {
            "available": True,
            "method": "Same-model text omission check",
            "fraud_score": original_score,
            "threshold": FRAUD_THRESHOLD,
            "review_required": HUMAN_REVIEW_LOWER_THRESHOLD < original_score < HUMAN_REVIEW_UPPER_THRESHOLD,
            "flagged": original_score >= FRAUD_THRESHOLD,
            "features": [],
            "message": "Add more listing text to compare what raises or lowers the model score.",
        }

    try:
        results = classifier(
            masked_texts,
            candidate_labels=CANDIDATE_LABELS,
            hypothesis_template="This listing is {}.",
            multi_label=True,
            batch_size=CLASSIFIER_BATCH_SIZE,
            truncation=True,
        )
        features = []
        for index, result in enumerate(results):
            if index >= len(segments):
                break
            category_scores_without_text = get_fraud_category_scores(result)
            score_without_text = max(category_scores_without_text.values())
            category_changes = {
                category: original_category_scores[category] - category_score
                for category, category_score in category_scores_without_text.items()
            }
            most_affected_category = max(
                category_changes, key=lambda category: abs(category_changes[category])
            )
            contribution = original_score - score_without_text
            features.append(
                {
                    "text": segments[index],
                    "score_without_text": round(score_without_text, 4),
                    "contribution": round(contribution, 4),
                    "category": most_affected_category,
                    "category_contribution": round(
                        category_changes[most_affected_category], 4
                    ),
                    "flag_without_text": score_without_text >= FRAUD_THRESHOLD,
                    "review_required_without_text": (
                        HUMAN_REVIEW_LOWER_THRESHOLD
                        < score_without_text
                        < HUMAN_REVIEW_UPPER_THRESHOLD
                    ),
                }
            )

        features.sort(key=lambda item: abs(item["contribution"]), reverse=True)
        risk_raising = [feature for feature in features if feature["contribution"] > 0]
        strongest_risk_signal = risk_raising[0] if risk_raising else None
        label, review_required, flagged = get_decision(original_score)
        if review_required:
            summary = (
                f"Human review is required: the score ({original_score:.1%}) is "
                f"above {HUMAN_REVIEW_LOWER_THRESHOLD:.0%} and below "
                f"{HUMAN_REVIEW_UPPER_THRESHOLD:.0%}. The classifier's "
                f"{FRAUD_THRESHOLD:.0%} warning cutoff is "
                f"{'crossed' if flagged else 'not crossed'}; do not treat this "
                "borderline result as an automatic decision."
            )
        elif flagged:
            summary = (
                f"The classifier flagged this listing ({original_score:.1%}), "
                f"which is at or above the {FRAUD_THRESHOLD:.0%} cutoff."
            )
        else:
            summary = (
                f"The classifier did not flag this listing ({original_score:.1%}); "
                f"it is below the {FRAUD_THRESHOLD:.0%} cutoff."
            )
        if strongest_risk_signal:
            summary += (
                f" The strongest score-raising excerpt was "
                f"\"{strongest_risk_signal['text']}\"; without it, the score "
                f"falls to {strongest_risk_signal['score_without_text']:.1%}. "
                f"It most affected the \"{strongest_risk_signal['category']}\" category."
            )
        else:
            summary += " No tested excerpt raised the fraud-risk score."

        return {
            "available": True,
            "method": "Same-model text omission check",
            "fraud_score": round(float(original_score), 4),
            "threshold": FRAUD_THRESHOLD,
            "review_required": review_required,
            "flagged": flagged,
            "features": features,
            "summary": summary,
        }
    except Exception as error:
        app.logger.warning("Model explanation failed: %s", error)
        return {
            "available": False,
            "method": "Same-model text omission check",
            "fraud_score": round(float(original_score), 4),
            "threshold": FRAUD_THRESHOLD,
            "review_required": HUMAN_REVIEW_LOWER_THRESHOLD < original_score < HUMAN_REVIEW_UPPER_THRESHOLD,
            "flagged": original_score >= FRAUD_THRESHOLD,
            "features": [],
            "message": "Could not compare text segments with the classifier.",
        }


@lru_cache(maxsize=1)
def get_classifier():
    from transformers import pipeline
    import torch

    return pipeline(
        "zero-shot-classification",
        model=MODEL_ID,
        device=0 if torch.cuda.is_available() else -1,
    )


def analyze_listing(text, classifier=None):
    classifier = classifier or get_classifier()
    result = classifier(
        text,
        candidate_labels=CANDIDATE_LABELS,
        hypothesis_template="This listing is {}.",
        multi_label=True,
        batch_size=CLASSIFIER_BATCH_SIZE,
        truncation=True,
    )
    scores = dict(zip(result["labels"], result["scores"]))
    fraud_category_scores = get_fraud_category_scores(result)
    fraud_score = max(fraud_category_scores.values())
    ranked_categories = sorted(
        scores.items(), key=lambda item: item[1], reverse=True
    )[:3]
    label, review_required, flagged = get_decision(fraud_score)
    ranked_categories = [
        {"label": category, "score": round(float(score), 4)}
        for category, score in ranked_categories
    ]

    analysis = {
        "label": label,
        "fraud_score": round(float(fraud_score), 4),
        "threshold": FRAUD_THRESHOLD,
        "human_review_thresholds": {
            "lower_exclusive": HUMAN_REVIEW_LOWER_THRESHOLD,
            "upper_exclusive": HUMAN_REVIEW_UPPER_THRESHOLD,
        },
        "review_required": review_required,
        "flagged": flagged,
        "ranked_categories": ranked_categories,
        "fraud_category_scores": fraud_category_scores,
        "model": MODEL_ID,
        "disclaimer": (
            "This uncalibrated score is not a probability or verified fraud "
            "finding. The explanation compares the same classifier's score "
            "when text excerpts are omitted; it is a local sensitivity check, "
            "not a transcript of the model's thoughts. Check the listing and "
            "decide for yourself. The warning label uses a provisional "
            f"{FRAUD_THRESHOLD:.0%} cutoff that has not been validated."
        ),
    }
    return analysis


@app.get("/")
def index():
    return render_template(
        "index.html",
        model_id=MODEL_ID,
        input_limit=MAX_TEXT_LENGTH,
        review_lower_percent=f"{HUMAN_REVIEW_LOWER_THRESHOLD:.0%}",
        review_upper_percent=f"{HUMAN_REVIEW_UPPER_THRESHOLD:.0%}",
        fraud_threshold_percent=f"{FRAUD_THRESHOLD:.0%}",
        job_retention_minutes=ANALYSIS_JOB_TTL_SECONDS // 60,
    )


@app.post("/api/analyze")
def api_analyze():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="Send a JSON object with a 'text' field."), 400

    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        return jsonify(error="'text' must be a non-empty string."), 400
    if len(text) > MAX_TEXT_LENGTH:
        return jsonify(error=f"'text' must be at most {MAX_TEXT_LENGTH} characters."), 400

    prune_analysis_jobs()
    with analysis_jobs_lock:
        if len(analysis_jobs) >= MAX_ANALYSIS_JOBS:
            return jsonify(error="The analysis queue is full. Try again shortly."), 503

        job_id = uuid4().hex
        future = analysis_executor.submit(analyze_listing, text.strip())
        analysis_jobs[job_id] = {
            "analysis_future": future,
            "explanation_future": None,
            "created_at": time.monotonic(),
            "text": text.strip(),
        }

    return jsonify(job_id=job_id, status="processing"), 202


@app.get("/api/analyze/<job_id>")
def api_analysis_status(job_id):
    prune_analysis_jobs()
    with analysis_jobs_lock:
        job = analysis_jobs.get(job_id)

    if job is None:
        return jsonify(error="Analysis job was not found or has expired."), 404

    future = job["analysis_future"]
    if not future.done():
        return jsonify(status="processing"), 202

    try:
        return jsonify(future.result())
    except Exception:
        app.logger.exception("Marketplace listing classification failed")
        return jsonify(error="The classification model is unavailable."), 503


@app.post("/api/analyze/<job_id>/explain")
def api_start_explanation(job_id):
    prune_analysis_jobs()
    with analysis_jobs_lock:
        job = analysis_jobs.get(job_id)
        if job is None:
            return jsonify(error="Analysis job was not found or has expired."), 404
        if not job["analysis_future"].done():
            return jsonify(error="Wait for the classification result first."), 409
        if job["explanation_future"] is None:
            analysis = job["analysis_future"].result()
            job["explanation_future"] = analysis_executor.submit(
                explain_with_classifier,
                job["text"],
                analysis["fraud_score"],
                analysis["fraud_category_scores"],
            )

    return jsonify(status="processing"), 202


@app.get("/api/analyze/<job_id>/explain")
def api_explanation_status(job_id):
    prune_analysis_jobs()
    with analysis_jobs_lock:
        job = analysis_jobs.get(job_id)
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
        app.logger.exception("Model explanation failed")
        return jsonify(error="The model explanation is unavailable."), 503


@app.post("/api/analyze/<job_id>/review")
def api_record_review(job_id):
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="Send a JSON object with a review decision."), 400

    decision = payload.get("decision")
    if not isinstance(decision, str) or decision not in REVIEW_DECISIONS:
        return jsonify(error="Choose a valid human review decision."), 400

    reviewer_id = payload.get("reviewer_id", "")
    rationale = payload.get("rationale", "")
    if not isinstance(reviewer_id, str) or len(reviewer_id) > 80:
        return jsonify(error="Reviewer ID must be at most 80 characters."), 400
    if not isinstance(rationale, str) or len(rationale) > 500:
        return jsonify(error="Rationale must be at most 500 characters."), 400

    with analysis_jobs_lock:
        job = analysis_jobs.get(job_id)
    if job is None:
        return jsonify(error="Analysis job was not found or has expired."), 404
    if not job["analysis_future"].done():
        return jsonify(error="Wait for the classification result first."), 409

    try:
        analysis = job["analysis_future"].result()
        review_id = record_human_review(
            job_id,
            analysis,
            decision,
            reviewer_id.strip() or None,
            rationale.strip() or None,
        )
    except Exception:
        app.logger.exception("Human review could not be recorded")
        return jsonify(error="The review could not be saved."), 503

    return jsonify(
        status="recorded",
        review_id=review_id,
        decision=decision,
        model_label=analysis["label"],
        fraud_score=analysis["fraud_score"],
    ), 201


if __name__ == "__main__":
    app.run(
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "5000")),
        debug=False,
    )
