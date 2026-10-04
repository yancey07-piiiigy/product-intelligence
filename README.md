# ShoeLens

ShoeLens is a local, evidence-linked shoe search prototype. A user uploads a JPG, PNG, or WebP photo and the system searches a fixed catalog of 2,000 shoes. It returns one of three outcomes:

1. **Accepted match** — a catalog identity is supported by the retrieval score and the gap over the second candidate.
2. **Uncertain shoe** — similar catalog items are shown as alternatives, without claiming an identity.
3. **Non-footwear refusal** — the image is rejected locally and is not sent to the optional vision API.

The product decision is made by local code. OpenRouter is optional and is used only to explain catalog fields for accepted matches or describe visible features for uncertain shoes. It does not decide product identity.

See [Product documentation](docs/PRODUCT.md) for the persona, inputs, outputs, architecture, target metrics, reached metrics, and product limitations. See the [data explainer](docs/DATA.md) and [evaluation explainer](docs/EVALUATION.md) for the checked-in data and evaluation evidence.

## Requirements

- macOS or Linux
- Python 3.12 recommended
- About 2 GB of free disk space for Python packages and the local CLIP model
- Internet access during initial dependency/model setup
- An OpenRouter API key only if AI summaries and uncertain-photo descriptions are required

The recorded evaluation used macOS arm64 and CPU inference. Exact Python packages are pinned in `requirements-lock.txt`.

## Quick start

On macOS, the simplest route is:

```bash
chmod +x setup.command start.command
./setup.command
./start.command
```

Then open `http://127.0.0.1:8501`.

Equivalent manual setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python server.py
```

The repository contains the prepared 2,000-product catalog, local index, frozen test queries, and evaluation outputs. If the CLIP model is not included in the submitted copy, obtain it during setup with:

```bash
.venv/bin/python prepare_project.py download
```

## Optional OpenRouter configuration

Copy the example configuration and add your own key:

```bash
cp .env.example .env
```

Set `OPENROUTER_API_KEY` in `.env`. The tested model is `openai/gpt-4o-mini`. Never commit or show `.env` in a demo recording. Without a key, local retrieval, identity gating, and non-footwear refusal still work; only remote text generation is unavailable.

## Run checks and evaluations

Run the Python tests:

```bash
.venv/bin/python -m unittest discover -s tests
```

Run the frontend logic test and production build:

```bash
cd frontend
node --test src/evaluationView.test.js
npm run build
cd ..
```

Recompute the frozen local benchmark and user-study analysis:

```bash
.venv/bin/python benchmark.py --study artifacts/user-study.csv
```

Refresh existing OpenRouter evaluation reports without making new paid calls:

```bash
.venv/bin/python evaluate_summary.py
.venv/bin/python evaluate_visual.py
```

Live API evaluation is opt-in. These commands can incur charges:

```bash
.venv/bin/python evaluate_summary.py --max-summaries 42
.venv/bin/python evaluate_visual.py --max-images 9
```

Details, metric definitions, result scope, and output files are documented in the [evaluation explainer](docs/EVALUATION.md).

## Repository guide

| Path | Purpose |
|---|---|
| `server.py` | Local HTTP service and API routing |
| `vision.py` | CLIP retrieval and product search |
| `geometry.py` | SIFT and affine-RANSAC geometric verification |
| `summary_service.py` | Source-constrained catalog summary generation |
| `vision_summary.py` | Optional visible-feature description for uncertain shoes |
| `frontend/` | React interface and evaluation views |
| `docs/` | Product, data, and evaluation documentation |
| `data/` | Submitted 2,000-product catalog data and study task definitions |
| `artifacts/` | Frozen queries, indexes, predictions, metrics, and review records |
| `tests/` | Python evaluation and behavior tests |
| `study/` | User-study instructions and task images |
| `demo/` | Demo slides, script, and three frozen example cases |

Generated environments and secrets are intentionally excluded by `.gitignore`, including `.env`, `.venv/`, `frontend/node_modules/`, runtime logs, model weights, and API response caches.

## Scope

ShoeLens is a course prototype for a fixed catalog. Its held-out test photos are controlled transformations of catalog-origin images, rather than independent phone photographs. It does not identify arbitrary products on the web, verify authenticity, or infer facts absent from the catalog. Consult the [evaluation explainer](docs/EVALUATION.md) before interpreting the reported accuracy.
