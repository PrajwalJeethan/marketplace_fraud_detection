# Marketplace Fraud Detection

## Run the text analysis app

The Flask app classifies a pasted listing description with the smaller Hugging
Face zero-shot model `typeform/distilbert-base-uncased-mnli` by default, to
reduce CPU inference time. The model is downloaded and initialized on the first
analysis request, so that request may take longer; subsequent requests reuse the
locally cached model. Set `HF_MODEL_ID` to use another compatible model. The
classifier batches label comparisons and uses CUDA when available.

The **Explain result** button compares the same classifier's score on the
original listing and on versions with up to three sentence/phrase groups
omitted. This lightweight sensitivity check indicates which excerpts raise or
lower the score and whether omitting one would flip the threshold decision. It
does not train a model or reveal hidden model thoughts. The smaller default
model may produce different scores, so evaluate it on representative examples
before drawing conclusions about accuracy.

```bash
python -m pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:5000>. The JSON API is `POST /api/analyze` with a body
such as `{"text": "Listing description..."}`. It queues analysis and returns
a job ID; poll `GET /api/analyze/<job_id>` for the uncalibrated `fraud_score`,
coarse label, and closest model-matched categories. The omission check runs
only when the user clicks **Explain result**. The API equivalent is
`POST /api/analyze/<job_id>/explain`, followed by polling
`GET /api/analyze/<job_id>/explain` for the excerpt score changes.

The prototype keeps the human in control: it offers a risk signal only and
does not take action on a listing. Reviewers can record a decision to confirm
the warning, mark the listing legitimate, or escalate it. Review metadata
(model, score, outcome, timestamp, and optional reviewer ID/rationale) is saved
in `instance/review_decisions.sqlite3`; raw listing text is not included. Set
`REVIEW_DB_PATH` to choose another local database path. The demo has no login or
reviewer authentication, so do not use it for real cases or sensitive
identifiers. Review records do not automatically retrain the model. The
separate Yes/No usefulness feedback remains in the browser page only.

Listing text is processed locally, not written to disk, and held in server
memory while its async job runs. Job records expire after 15 minutes and are
pruned on the next request; without another request, they may remain until the
app restarts. Avoid entering personal information.

Scores strictly above 30% and below 70% are labeled `human_review_required`;
scores at or above 70% are `potentially_fraudulent`. Scores at or below 30% are
`likely_legitimate`. These thresholds are
provisional and have not been calibrated on representative marketplace data.

For Task 1, the design follows a human-centred AI approach: the model provides
an assistive signal, while the user retains the decision. It makes the model
name, candidate-label matches, provisional cutoff, and limitations visible.
This is a prototype design rationale, not a claim of formal certification.

A `GET /favicon.ico` 404 in the server log is a separate browser request for a
site icon; it does not indicate that the analysis API failed. The analysis
endpoint accepts `POST` requests, not `GET` requests.

This general-purpose model is not trained specifically on marketplace fraud.
Its score is not a calibrated probability or a measure of accuracy, and it can
make mistakes. Treat results as a first-pass signal, not proof; evaluate and
fine-tune on representative, labeled marketplace listings before relying on
automated decisions. For the assignment evaluation, test with 3-5 users or user
representatives, ask whether they understand the output and feel in control,
and record their actual feedback. Do not present model scores or this app's
automated tests as evidence of user trust or real-world accuracy.

## Get marketplace scam data

The closest Hugging Face match for Facebook Marketplace-style scam detection is
[`useKled/marketplace-scam-listings`](https://huggingface.co/datasets/useKled/marketplace-scam-listings).
It describes labeled screenshots of suspected scam listings from Facebook
Marketplace, Shopee, Lazada, and similar platforms. It is not eBay-specific.

This dataset is gated, and its Hugging Face card states that the data files are
delivered by the provider rather than stored in the repository. Request access
on the dataset page and request a sample from the provider; Hugging Face access
alone may not provide the data files.

Install the downloader dependency and authenticate with an account that has
access:

```bash
python -m pip install -r requirements.txt
hf auth login
python scripts/download_marketplace_data.py
```

Alternatively, set `HF_TOKEN` to a Hugging Face access token. The script
downloads available repository files to `data/marketplace-scam-listings` and
reports clearly if the repository contains only its dataset card. If the
provider supplies the data separately, place those files in that directory.
