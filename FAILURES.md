# Failure modes, fixes, and results

Honest engineering notes for this project. Nothing here is invented for polish.

## What can go wrong

- **Leaky time or project splits when training delay risk models.** Impact: over-optimistic PMO forecasts. Mitigation: keep training code (`train.py`) and serving (`predict.py` / FastAPI `app.py`) explicit; review split logic before trusting scores.
- **SHAP or model artifacts missing at runtime.** Impact: UI explanation fails. Mitigation: `models/` directory ships with the app; fail clearly if artifacts are absent.
- **Schema drift between training features and API payload.** Impact: silent bad scores. Mitigation: shared `schemas.py` / `config.py` for request and feature contracts.
- **No README onboarding.** Impact: operators cannot reproduce the demo. Mitigation: README.md added on 1 November 2026 with a verified quick start.

## What went wrong

**No recorded production incident in this repo yet.** History is thin: initial commit of FastAPI + XGBoost + SHAP UI, then MIT license. There is **no README.md** in the tree at documentation time, so operational failures are not narrated in docs.

## How it was resolved

- Serving path is FastAPI with static UI assets; training and prediction are separate modules so model rebuilds do not require editing the API surface by hand.
- Further split/leakage hardening should be verified in `train.py` before any production claim.

## Results

- Successful local demo (from layout): install `requirements.txt`, run training if needed, start `app.py`, exercise the SHAP-enabled UI against the bundled model.
- No published accuracy, MAE, or latency figures are in this repository.
