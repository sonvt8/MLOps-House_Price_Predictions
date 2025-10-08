# MLOps-House_Price_Predictions

End-to-end workflow for house price prediction: data processing, feature engineering, model training with MLflow tracking, API serving with FastAPI, and Streamlit UI.

Course reference: https://www.udemy.com/course/devops-to-mlops-bootcamp

## Prerequisites
- Python 3.11, pip
- Docker and Docker Compose
- Optional: virtual environment (recommended)

## Project structure (key paths)
- `data/raw/house_data.csv`
- `data/processed/cleaned_data.csv`
- `data/featured/featured_house_data.csv`
- `src/processing/data_processing.py` (cleaning CLI)
- `src/e_featuring/engineer.py` (feature engineering CLI)
- `src/configs/model_config.yaml` (training config)
- `src/training/train_model.py` (train + CV + MLflow logging)
- `src/models/trained/` (saved artifacts)
- `src/api/main.py` (FastAPI app), `src/api/run_api.py` (dev launcher)
- `src/streamlit_app/app.py` (UI)
- `deployment/mlflow/docker-compose.yaml` (MLflow tracking server)

---

## Stage 1 — Data → Features → Train (with MLflow)

1) Start MLflow Tracking Server

```bash
docker compose -f deployment/mlflow/docker-compose.yaml up -d
# MLflow UI: http://localhost:5555
```

2) Clean raw data → `data/processed/cleaned_data.csv`

```bash
python src/processing/data_processing.py \
  --input data/raw/house_data.csv \
  --output data/processed
```

3) Feature engineering → `data/featured/featured_house_data.csv`

```bash
python src/e_featuring/engineer.py \
  --input data/processed/cleaned_data.csv \
  --output data/featured/featured_house_data.csv
```

4) Train model (GridSearchCV inside Pipeline; logs to MLflow)

```bash
python src/training/train_model.py \
  --config src/configs/model_config.yaml \
  --data data/featured/featured_house_data.csv \
  --models-dir src/models/trained \
  --mlflow-tracking-uri http://localhost:5555 \
  --experiment-name "HousePrice - Experiments"
```

Outputs:
- Artifacts: `src/models/trained/model_pipeline.joblib`, `feature_names.json`, `metrics.json`
- MLflow run with params/metrics/artifacts under the configured tracking server

---

## Stage 2 — Serve API and Streamlit UI

1) Start FastAPI server (development)

```bash
# Option A: helper script (prints friendly messages)
python src/api/run_api.py

# Option B: uvicorn directly
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Quick checks:
```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "sqft": 1500,
        "bedrooms": 3,
        "bathrooms": 2,
        "location": "Urban",
        "year_built": 2000,
        "condition": "Good"
      }'
```

2) Start Streamlit UI

```bash
# Optionally point UI to a remote API
export API_URL=http://localhost:8000

streamlit run src/streamlit_app/app.py
```

Then open the browser tab that Streamlit prints, fill the form, and click “Predict Price”.

---

## Stage 3 — Dockerize the API and Streamlit UI (continuing update)

Build the API image:
```bash
docker build -t house-price-api .
```

Run the container:
```bash
docker run -d --rm -p 8000:8000 \
  -e APP_VERSION=1.0.0 \
  house-price-api
```

Optional: mount local model artifacts (for rapid iteration):
```bash
docker run --rm -p 8000:8000 \
  -v $(pwd)/src/models/trained:/app/src/models/trained \
  house-price-api
```

Build the Streamlit UI image:
```bash
docker build -f src/streamlit_app/Dockerfile -t house-price-ui .
```

Run the Streamlit UI container (pointing to local API):
```bash
docker run -d --rm -p 8501:8501 \
  -e API_URL=http://host.docker.internal:8000 \
  -e APP_VERSION=1.0.0 \
  house-price-ui
```

Next steps (continuing update):
- Containerize Streamlit and/or compose UI + API together
- CI/CD for training and serving
- Model registry and promotion flows with MLflow
- Cloud deployment

---

## Stage 4 — Publish Docker Images to Docker Hub

Assume your Docker Hub username is `YOUR_DOCKERHUB_USER` and you want to publish:
- API image built in Stage 3 as `house-price-api`
- Streamlit UI image built from `src/streamlit_app/Dockerfile` as `house-price-ui`

1) Login
```bash
docker login
```

2) Tag images (replace `YOUR_DOCKERHUB_USER` and version `v1` as desired)
```bash
# API
docker tag house-price-api YOUR_DOCKERHUB_USER/house-price-api:v1
docker tag house-price-api YOUR_DOCKERHUB_USER/house-price-api:latest

# UI
docker tag house-price-ui YOUR_DOCKERHUB_USER/house-price-ui:v1
docker tag house-price-ui YOUR_DOCKERHUB_USER/house-price-ui:latest
```

3) Push images
```bash
docker push YOUR_DOCKERHUB_USER/house-price-api:v1
docker push YOUR_DOCKERHUB_USER/house-price-api:latest

docker push YOUR_DOCKERHUB_USER/house-price-ui:v1
docker push YOUR_DOCKERHUB_USER/house-price-ui:latest
```

Quick pull/run test from another machine:
```bash
docker run -d --rm -p 8000:8000 YOUR_DOCKERHUB_USER/house-price-api:latest
docker run -d --rm -p 8501:8501 -e API_URL=http://host.docker.internal:8000 YOUR_DOCKERHUB_USER/house-price-ui:latest
```

---

## Stage 5 — Run MLflow + API + Streamlit via Docker Compose

A unified compose lives at `src/docker-compose.yaml`. Because it mounts MLflow volumes using paths relative to the project root, run the following commands from the `src/` directory.

1) Start all services
```bash
cd src
docker compose up -d --build
# MLflow     -> http://localhost:5555
# API        -> http://localhost:8000 (health: /health, docs: /docs)
# Streamlit  -> http://localhost:8501
```

2) Tail logs
```bash
docker compose logs -f mlflow
docker compose logs -f api
docker compose logs -f streamlit
```

3) Stop and remove containers
```bash
docker compose down --volumes
```

Notes:
- The compose mounts model artifacts from `src/models/trained` into the API container for rapid iteration.
- MLflow metadata and artifacts are persisted under `deployment/mlflow/mlflow_db` and `deployment/mlflow/mlruns` on the host.
