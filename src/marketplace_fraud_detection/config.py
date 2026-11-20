import os


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
FEEDBACK_VALUES = {"helpful", "not-helpful"}
