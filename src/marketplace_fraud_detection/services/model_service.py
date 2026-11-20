import logging
import re
from functools import lru_cache

from ..config import (
    CANDIDATE_LABELS,
    CLASSIFIER_BATCH_SIZE,
    FRAUD_LABELS,
    FRAUD_THRESHOLD,
    HUMAN_REVIEW_LOWER_THRESHOLD,
    HUMAN_REVIEW_UPPER_THRESHOLD,
    MAX_EXPLANATION_SEGMENTS,
    MODEL_ID,
)


logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_classifier():
    from transformers import pipeline
    import torch

    return pipeline(
        "zero-shot-classification",
        model=MODEL_ID,
        device=0 if torch.cuda.is_available() else -1,
    )


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

    return {
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


def explain_with_classifier(
    text, original_score, original_category_scores, classifier=None
):
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
            "review_required": (
                HUMAN_REVIEW_LOWER_THRESHOLD
                < original_score
                < HUMAN_REVIEW_UPPER_THRESHOLD
            ),
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
                category_changes,
                key=lambda category: abs(category_changes[category]),
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
        logger.warning("Model explanation failed: %s", error)
        return {
            "available": False,
            "method": "Same-model text omission check",
            "fraud_score": round(float(original_score), 4),
            "threshold": FRAUD_THRESHOLD,
            "review_required": (
                HUMAN_REVIEW_LOWER_THRESHOLD
                < original_score
                < HUMAN_REVIEW_UPPER_THRESHOLD
            ),
            "flagged": original_score >= FRAUD_THRESHOLD,
            "features": [],
            "message": "Could not compare text segments with the classifier.",
        }
