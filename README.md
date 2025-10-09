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

## Stage 0 — Manual Testing with Google Colab (Optional)

Before running the automated pipeline, you can manually test the data processing and model experimentation using Jupyter notebooks on Google Colab. This helps validate the approach and generate configuration files.

### Prerequisites for Colab
- Google account
- Access to Google Colab
- NGROK_AUTH_TOKEN (for MLflow UI access in Colab)

### Step 1: Upload Data to Google Drive
1. Upload `data/raw/house_data.csv` to your Google Drive
2. Place it in a folder like `Colab Notebooks/House Pricing/`

### Step 2: Run Feature Engineering Notebook
1. Open `notebooks/02_feature_engineering.ipynb` in Google Colab
2. Mount Google Drive and update the data path if needed
3. Run all cells to:
   - Create features: `house_age`, `price_per_sqft`, `bed_bath_ratio`, `total_rooms`
   - Generate `featured_house_data.csv` in your Google Drive

### Step 3: Run Experimentation Notebook
1. Open `notebooks/03_experimentation.ipynb` in Google Colab
2. Set up NGROK_AUTH_TOKEN in Colab Secrets (for MLflow UI access)
3. Run all cells to:
   - Compare multiple ML models (LinearRegression, RandomForest, GradientBoosting, XGBoost)
   - Generate `model_config.yaml` with best model configuration
   - View MLflow UI via ngrok tunnel

### Step 4: Download Generated Files
After running the notebooks, download these files to your local project:
- `featured_house_data.csv` → place in `data/featured/`
- `model_config.yaml` → place in `src/configs/`

### Benefits of Manual Testing
- ✅ Validate data processing steps
- ✅ Experiment with different models
- ✅ Generate optimal configuration
- ✅ Understand the MLflow tracking process
- ✅ Create baseline for automated pipeline

### Troubleshooting Google Colab
- **NGROK_AUTH_TOKEN**: Get free token from https://ngrok.com/ and add to Colab Secrets
- **File paths**: Update paths in notebooks if your Google Drive folder structure differs
- **Package conflicts**: Restart runtime if you encounter import errors
- **MLflow UI access**: Use the ngrok URL provided in the notebook output

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

**Note:** If you completed Stage 0 (Google Colab), you can use the generated `model_config.yaml` and `featured_house_data.csv` files directly, or let the pipeline regenerate them with the same configuration.

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

---

## CI — GitHub Actions validation (Stages 1→5, no image build/push)

This repository includes an automated validation workflow that re-runs the key stages to ensure the current codebase is healthy after each change to `main`.

Where: `.github/workflows/ci-validate.yml`

What it does:
- Starts MLflow tracking via compose and waits for `http://localhost:5555` to be available
- Runs Stage 1 locally: data cleaning, feature engineering, and model training (logs to MLflow)
- Stops that MLflow instance to avoid container name conflicts
- Starts the full stack from `src/docker-compose.yaml` (MLflow + API + Streamlit) using prebuilt Docker Hub images
- Prints container logs and runs health checks:
  - API `/health` + quick docs and a smoke `POST /predict`
  - Streamlit root (HTTP 200 check)
- Tears down all containers and volumes

Triggers:
- Push to `main`
- Manual trigger via Actions tab (workflow_dispatch)

Requirements:
- Docker is available on the runner (provided by `ubuntu-latest`)
- Public images exist in Docker Hub:
  - `sonvt8/house-price-api:latest`
  - `sonvt8/house-price-ui:latest`

How to run (manual):
1. Open GitHub → Actions → "CI Validate Pipeline"
2. Click "Run workflow"
3. Select branch (e.g., `main`) → Run

How to inspect results:
- Open the workflow run → check steps:
  - "Tail container logs (short)" prints logs for `mlflow`, `api`, and `streamlit`
  - "Health check API" and "Health check Streamlit" verify endpoints
  - Failure at any step will mark the workflow red with error context

Customize image names (for forks):
- If you publish images under a different Docker Hub namespace, update the image references in the step "Prepare compose override to use prebuilt images" inside `.github/workflows/ci-validate.yml`.

MLOps role of this workflow:
- Continuous validation: Ensures the end-to-end pipeline (data → features → training → serving) remains functional on every push to production (`main`).
- Early failure signal: Surfaces broken dependencies or regressions via container logs and health checks.
- Separation of concerns: Image build/publish is handled in a dedicated workflow; this job focuses on runtime validation using prebuilt images.

---

## CI — Docker image build & publish to Docker Hub

Where: `.github/workflows/docker-publish.yml`

What it does:
- Builds multi-step matrix of Docker images for:
  - `house-price-api` from project `Dockerfile`
  - `house-price-ui` from `src/streamlit_app/Dockerfile`
- Logs in to Docker Hub
- Tags images from the Git tag that triggered the workflow
- Pushes images to `DOCKERHUB_USERNAME/<image>:<tag>`

Triggers:
- Push of a Git tag matching `v*.*.*` (e.g., `v1.0.0`)

Required GitHub Secrets (Repository → Settings → Secrets and variables → Actions):
- `DOCKERHUB_USERNAME`: Your Docker Hub username (e.g., `sonvt8`)
- `DOCKERHUB_TOKEN`: A Docker Hub access token or password

How to create a Docker Hub token:
1. Go to Docker Hub → Account Settings → Security → New Access Token
2. Give it a name (e.g., `github-ci`) and copy the generated token
3. Save it as `DOCKERHUB_TOKEN` in your GitHub repository secrets

How to release images:
1. Create a Git tag locally (example `v1.0.0`) and push it:
   ```bash
   git tag v1.0.0
   git push origin v1.0.0
   ```
2. The workflow will build and push:
   - `${DOCKERHUB_USERNAME}/house-price-api:v1.0.0`
   - `${DOCKERHUB_USERNAME}/house-price-ui:v1.0.0`

Notes:
- The workflow uses `docker/metadata-action` to generate tags from the Git tag.
- It currently pushes for `linux/amd64` (adjust `platforms` if needed).
- Use the published images in the validation workflow and local compose by referencing your Docker Hub namespace.

MLOps role of this workflow:
- Reproducible artifacts: Produces versioned, portable container images for API/UI.
- Promotion & release: Git-tag driven releases align application versions with image tags.
- Separation of concerns: Keeps build/publish independent from runtime validation.

---