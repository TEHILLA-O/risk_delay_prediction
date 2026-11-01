# PMO Delay Risk

A small FastAPI app, for project managers and anyone learning explainable ML, that scores how likely a project is to run late from five inputs and explains the score in plain English using SHAP.

The model is trained on synthetic data generated from a formula in `train.py`, not on real projects. Treat the output as a demo of the method, not as a forecast.

See [FAILURES.md](FAILURES.md) for what can go wrong and what has been fixed.

## What it does

- Takes five project features: team size, days to deadline, budget, task complexity and number of dependencies.
- `POST /predict` returns the probability of delay.
- `POST /explain` also returns a SHAP contribution per feature and a one sentence explanation, for example "This project has a 92% delay risk due to a tight deadline, constrained budget, and limited team capacity."
- A single page UI at `/` lets you enter the numbers and see the score and the chart.

## How it works

1. `train.py` generates 5,000 synthetic projects. Delay risk rises with tight deadlines, small teams, low budget, high complexity and many dependencies, plus random noise. You can pass your own CSV with `--data` instead.
2. It trains an XGBoost classifier on an 80/20 split and saves the model plus 100 background rows to `models/delay_risk_xgb.joblib`.
3. `app.py` loads that file at startup, builds a SHAP explainer on the predicted delay probability, and serves the API and the static UI.
4. `predict.py` turns the largest positive SHAP values into the sentence in the response.

## Quick start

Needs Python 3.10 or later.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --port 8000
```

Open http://localhost:8000. A trained model is already in `models/`, so training is optional. To retrain:

```bash
python train.py                      # synthetic data, writes models/delay_risk_xgb.joblib
python train.py --data my_projects.csv --output models/my_model.joblib
```

A CSV needs the columns `team_size`, `deadline_days`, `budget`, `task_complexity`, `dependencies_count` and `delay_risk` (0 or 1). The app always loads `models/delay_risk_xgb.joblib`.

## Example

```bash
curl -s localhost:8000/explain -H "Content-Type: application/json" \
  -d '{"team_size":4,"deadline_days":20,"budget":50000,"task_complexity":8,"dependencies_count":15}'
```

Real response from the bundled model:

```json
{"delay_risk_probability":0.9233653545379639,"delay_risk_percent":92,
 "explanation":"This project has a 92% delay risk due to a tight deadline, constrained budget, and limited team capacity.",
 "shap_values":{"team_size":0.1215,"deadline_days":0.3970,"budget":0.1537,"task_complexity":0.0800,"dependencies_count":0.0096},
 "feature_labels":{"team_size":"Team size","deadline_days":"Days to deadline","budget":"Budget","task_complexity":"Task complexity","dependencies_count":"Dependencies"}}
```

(SHAP values rounded here for reading.) A large, well funded project with a long deadline (20 people, 150 days, 1,500,000 budget, complexity 2, 2 dependencies) scored 0%.

## Verified run (1 November 2026, Linux, Python 3.12)

| Check | Result |
| --- | --- |
| Fresh venv and `pip install -r requirements.txt` | ok (xgboost 3.4.1, shap 0.54.1, scikit-learn 1.9.1) |
| ruff (errors and undefined names) | clean |
| Start with the bundled model | ok; XGBoost warns that the pickled model came from an older version |
| `/health`, `/` | 200 |
| `/predict`, `/explain` | 200, responses above |
| Invalid input (negative team size) | 422 |
| `python train.py` (to a scratch path) | holdout ROC AUC 0.8508, accuracy 0.86, recall for "delayed" 0.28 |
| Tests | none exist |
| CI | none |

The training scores are on synthetic data drawn from the same formula the model is learning, so they say nothing about real projects. Accuracy on real project data has not been measured.

## Structure

| File | What it is |
| --- | --- |
| `app.py` | FastAPI app: `/`, `/health`, `/predict`, `/explain` |
| `predict.py` | model loading, SHAP values and the explanation sentence |
| `train.py` | synthetic data generator and training script |
| `config.py` | feature names, labels, model path and explanation phrases |
| `schemas.py` | request and response models |
| `static/` | the single page UI |
| `models/delay_risk_xgb.joblib` | trained model bundle |

## Configuration

There are no environment variables. Feature names and the model path are in `config.py`; the port is whatever you pass to `uvicorn`.

## Limitations

- Synthetic training data only. Real project delay depends on much more than five numbers.
- Budget has no currency or scale; it only means something relative to the training data.
- The bundled model was pickled with an older XGBoost. It loads with a warning; retrain if you upgrade.
- Dependencies are not pinned, so results can drift between installs.
- No tests, no CI, no authentication.

## Status

A working demo of XGBoost with SHAP explanations. It has no real users or data.

## Licence

MIT, see [LICENSE](LICENSE).

More of my work: https://cv.omnitest.dev/
