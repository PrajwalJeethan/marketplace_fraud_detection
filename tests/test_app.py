import os
import sqlite3
import tempfile
import unittest
from concurrent.futures import Future
from unittest.mock import patch

import app as marketplace_app


class ImmediateExecutor:
    def submit(self, function, *args, **kwargs):
        future = Future()
        try:
            future.set_result(function(*args, **kwargs))
        except Exception as error:
            future.set_exception(error)
        return future


class MarketplaceAppTests(unittest.TestCase):
    def setUp(self):
        marketplace_app.app.config["TESTING"] = True
        self.client = marketplace_app.app.test_client()
        with marketplace_app.analysis_jobs_lock:
            marketplace_app.analysis_jobs.clear()
        self.executor_patch = patch.object(
            marketplace_app, "analysis_executor", ImmediateExecutor()
        )
        self.executor_patch.start()
        self.review_directory = tempfile.TemporaryDirectory()
        self.review_db_patch = patch.object(
            marketplace_app,
            "REVIEW_DB_PATH",
            os.path.join(self.review_directory.name, "review_decisions.sqlite3"),
        )
        self.review_db_patch.start()

    def tearDown(self):
        self.review_db_patch.stop()
        self.review_directory.cleanup()
        self.executor_patch.stop()

    def analyze(self, text):
        queued = self.client.post("/api/analyze", json={"text": text})
        self.assertEqual(queued.status_code, 202)
        self.assertEqual(queued.json["status"], "processing")
        return self.client.get(f"/api/analyze/{queued.json['job_id']}")

    def test_homepage_renders_listing_form(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Marketplace Listing Check", response.data)
        self.assertIn(b"not written to disk", response.data)
        self.assertIn(b"data-feedback", response.data)
        self.assertIn(b"Explain result", response.data)
        self.assertIn(b"Human review required", response.data)
        self.assertIn(b"Transparency and data use", response.data)
        self.assertIn(marketplace_app.MODEL_ID.encode(), response.data)
        self.assertIn(b"held in server memory", response.data)
        self.assertIn(b"Human review decision", response.data)
        self.assertIn(b"Record human decision", response.data)
        self.assertNotIn(b"SHAP", response.data)

    def test_explanation_segmenter_does_not_cache_listing_text(self):
        self.assertFalse(hasattr(marketplace_app.split_explanation_segments, "cache_info"))

    def test_analysis_returns_suspicious_label_and_score(self):
        mock_classifier = lambda *args, **kwargs: {
            "labels": marketplace_app.CANDIDATE_LABELS,
            "scores": [0.82, 0.74, 0.12, 0.08, 0.19, 0.15, 0.23],
        }
        with patch.object(marketplace_app, "get_classifier", return_value=mock_classifier):
            response = self.analyze("Pay outside the platform; urgent cash only.")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["label"], "potentially_fraudulent")
        self.assertEqual(response.json["fraud_score"], 0.82)
        self.assertEqual(response.json["threshold"], marketplace_app.FRAUD_THRESHOLD)
        self.assertEqual(
            response.json["fraud_category_scores"]["advance payment scam"], 0.74
        )
        self.assertIn("same classifier", response.json["disclaimer"])
        self.assertNotIn("explanation", response.json)
        self.assertNotIn("model_influence", response.json)
        self.assertEqual(len(response.json["ranked_categories"]), 3)
        self.assertEqual(
            response.json["ranked_categories"][0]["label"],
            marketplace_app.CANDIDATE_LABELS[0],
        )
        self.assertNotIn("confidence", response.json)

    def test_analysis_returns_legitimate_label(self):
        mock_classifier = lambda *args, **kwargs: {
            "labels": marketplace_app.CANDIDATE_LABELS,
            "scores": [0.12, 0.08, 0.05, 0.11, 0.09, 0.81, 0.78],
        }
        with patch.object(marketplace_app, "get_classifier", return_value=mock_classifier):
            response = self.analyze("Collection only, payment on pickup.")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["label"], "likely_legitimate")
        self.assertEqual(response.json["fraud_score"], 0.12)
        self.assertNotIn("explanation", response.json)

    def test_human_review_band_is_exclusive_at_30_and_70_percent(self):
        cases = (
            (0.30, "likely_legitimate", False),
            (0.31, "human_review_required", True),
            (0.50, "human_review_required", True),
            (0.69, "human_review_required", True),
            (0.70, "potentially_fraudulent", False),
            (0.71, "potentially_fraudulent", False),
        )
        for score, expected_label, expected_review in cases:
            with self.subTest(score=score):
                mock_classifier = lambda *args, **kwargs: {
                    "labels": marketplace_app.CANDIDATE_LABELS,
                    "scores": [score, 0.1, 0.1, 0.1, 0.1, 0.9, 0.8],
                }
                with patch.object(
                    marketplace_app, "get_classifier", return_value=mock_classifier
                ):
                    response = self.analyze("A marketplace listing description.")

                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json["threshold"], 0.7)
                self.assertEqual(response.json["label"], expected_label)
                self.assertEqual(response.json["review_required"], expected_review)

    def test_model_explanation_runs_only_after_explain_is_requested(self):
        mock_classifier = lambda *args, **kwargs: {
            "labels": marketplace_app.CANDIDATE_LABELS,
            "scores": [0.82, 0.74, 0.12, 0.08, 0.19, 0.15, 0.23],
        }
        explanation = {
            "available": True,
            "method": "Same-model text omission check",
            "fraud_score": 0.82,
            "features": [{"text": "Pay a deposit.", "contribution": 0.5}],
        }
        listing = "Pay a deposit. Collection only."
        category_scores = dict(zip(marketplace_app.FRAUD_LABELS, [0.82, 0.74, 0.12, 0.08, 0.19]))

        with (
            patch.object(marketplace_app, "get_classifier", return_value=mock_classifier),
            patch.object(marketplace_app, "explain_with_classifier", return_value=explanation) as explain,
        ):
            queued = self.client.post("/api/analyze", json={"text": listing})
            result = self.client.get(f"/api/analyze/{queued.json['job_id']}")

            self.assertEqual(result.status_code, 200)
            self.assertNotIn("explanation", result.json)
            explain.assert_not_called()

            started = self.client.post(f"/api/analyze/{queued.json['job_id']}/explain")
            explanation_response = self.client.get(
                f"/api/analyze/{queued.json['job_id']}/explain"
            )

        self.assertEqual(started.status_code, 202)
        self.assertEqual(explanation_response.status_code, 200)
        self.assertEqual(explanation_response.json["explanation"], explanation)
        explain.assert_called_once_with(listing, 0.82, category_scores)

    def test_human_review_persists_decision_without_listing_text(self):
        listing = "Private listing text is not part of the review record."
        mock_classifier = lambda *args, **kwargs: {
            "labels": marketplace_app.CANDIDATE_LABELS,
            "scores": [0.55, 0.2, 0.1, 0.1, 0.1, 0.15, 0.1],
        }
        with patch.object(marketplace_app, "get_classifier", return_value=mock_classifier):
            queued = self.client.post("/api/analyze", json={"text": listing})
            analysis_response = self.client.get(
                f"/api/analyze/{queued.json['job_id']}"
            )
            review_response = self.client.post(
                f"/api/analyze/{queued.json['job_id']}/review",
                json={
                    "decision": "marked_legitimate",
                    "reviewer_id": "reviewer-01",
                    "rationale": "Manual review found no suspicious payment request.",
                },
            )

        self.assertEqual(analysis_response.status_code, 200)
        self.assertEqual(review_response.status_code, 201)
        self.assertEqual(review_response.json["status"], "recorded")
        self.assertEqual(review_response.json["model_label"], "human_review_required")
        with sqlite3.connect(marketplace_app.REVIEW_DB_PATH) as connection:
            columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(human_reviews)")
            }
            record = connection.execute(
                "SELECT review_decision, reviewer_id, rationale FROM human_reviews"
            ).fetchone()

        self.assertNotIn("listing_text", columns)
        self.assertNotIn("text", columns)
        self.assertEqual(
            record,
            (
                "marked_legitimate",
                "reviewer-01",
                "Manual review found no suspicious payment request.",
            ),
        )

    def test_invalid_human_review_decision_is_rejected(self):
        response = self.client.post(
            "/api/analyze/not-a-job/review",
            json={"decision": "auto_delete_listing"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json)

    def test_text_omission_identifies_score_raising_excerpt(self):
        def mock_classifier(texts, **kwargs):
            if isinstance(texts, str):
                texts = [texts]
            results = []
            for text in texts:
                score = 0.9 if "pay a deposit" in text.lower() else 0.2
                results.append({
                    "labels": marketplace_app.CANDIDATE_LABELS,
                    "scores": [score, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2],
                })
            return results

        explanation = marketplace_app.explain_with_classifier(
            "Pay a deposit. Collection only.",
            0.9,
            dict(zip(marketplace_app.FRAUD_LABELS, [0.9, 0.1, 0.1, 0.1, 0.1])),
            mock_classifier,
        )

        self.assertTrue(explanation["available"])
        self.assertEqual(explanation["features"][0]["text"], "Pay a deposit.")
        self.assertEqual(explanation["features"][0]["score_without_text"], 0.2)
        self.assertEqual(explanation["features"][0]["contribution"], 0.7)
        self.assertFalse(explanation["features"][0]["flag_without_text"])
        self.assertIn("The classifier flagged this listing", explanation["summary"])
        self.assertIn('"Pay a deposit."', explanation["summary"])

    def test_analysis_status_is_processing_while_job_runs(self):
        pending_future = Future()
        with patch.object(marketplace_app, "analysis_executor") as executor:
            executor.submit.return_value = pending_future
            queued = self.client.post("/api/analyze", json={"text": "A listing."})

        self.assertEqual(queued.status_code, 202)
        status = self.client.get(f"/api/analyze/{queued.json['job_id']}")
        self.assertEqual(status.status_code, 202)
        self.assertEqual(status.json["status"], "processing")

    def test_empty_text_is_rejected(self):
        response = self.client.post("/api/analyze", json={"text": "  "})

        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json)

    def test_overlong_text_is_rejected(self):
        response = self.client.post(
            "/api/analyze",
            json={"text": "x" * (marketplace_app.MAX_TEXT_LENGTH + 1)},
        )

        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
