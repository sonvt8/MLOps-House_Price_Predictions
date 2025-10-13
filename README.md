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
uvicorn api.main:app --host 0.0.0.0 --port 8005
```

Quick checks:
```bash
curl http://localhost:8005/health

curl -X POST http://localhost:8005/predict \
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
export API_URL=http://localhost:8005

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
docker run -d --rm -p 8005:8005 \
  -e APP_VERSION=1.0.0 \
  house-price-api
```

Optional: mount local model artifacts (for rapid iteration):
```bash
docker run --rm -p 8005:8005 \
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
  -e API_URL=http://host.docker.internal:8005 \
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
docker run -d --rm -p 8005:8005 YOUR_DOCKERHUB_USER/house-price-api:latest
docker run -d --rm -p 8501:8501 -e API_URL=http://host.docker.internal:8005 YOUR_DOCKERHUB_USER/house-price-ui:latest
```

---

## Stage 5 — Run MLflow + API + Streamlit via Docker Compose

A unified compose lives at `src/docker-compose.yaml`. Because it mounts MLflow volumes using paths relative to the project root, run the following commands from the `src/` directory.

1) Start all services
```bash
cd src
docker compose up -d --build
# MLflow     -> http://localhost:5555
# API        -> http://localhost:8005 (health: /health, docs: /docs)
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

---

# Kubernetes Lab  — Local Environment & Kustomize Deployment (Post‑CI)

This section shows how to spin up a local multi‑node Kubernetes environment with **KIND**, validate it with **kube-ops-view**, and deploy the **API** + **Streamlit UI** using **Kustomize** manifests. It follows the steps from the K801 lab and adapts them to this repository structure.

## A. Prereqs (Windows 11 + Docker Desktop + WSL2)

- **Docker Desktop** enabled with **WSL2 backend** (recommended on Windows Home/Pro).
- **kubectl** in PATH (see earlier section of this README).
- **kind** installed (Windows binary or via Chocolatey: `choco install kind`).
- Optional but recommended: Windows Terminal / PowerShell 7.

Quick checks:
```powershell
docker version
kubectl version --client
kind version
```

> KIND uses Docker containers as Kubernetes **nodes**. Your “3 nodes” in K801 are 3 Docker containers joined into one cluster/network.

## B. Create a 3‑node KIND cluster

> **Get the helper files first (required for this step):**
```bash
git clone https://github.com/initcron/k8s-code.git
cd k8s-code/helper/kind/
```


From the repo root, if you have the file `kind-three-node-cluster.yaml` (from lab  [K801](https://kubernetes-tutorial.schoolofdevops.com/adv_kubernetes-setup/#)):

```powershell
kind create cluster --config kind-three-node-cluster.yaml
kubectl cluster-info --context kind-kind
kubectl get nodes
kubectl get pods -A
```

You should see one control‑plane and two workers (all **Ready**) and system pods like `kube-proxy`, `CoreDNS`, `kindnet`, `local-path-provisioner`, and static pods for the control plane.

> Tip (optional): To make NodePorts reachable on `localhost`, add `extraPortMappings` to your kind config for the ports you plan to use (e.g., 30000, 30100). Otherwise, use `kubectl port-forward` or the node container’s IP to access services.

## C. Visualize cluster with kube-ops-view

```powershell
git clone https://github.com/schoolofdevops/kube-ops-view
kubectl apply -f kube-ops-view/deploy/
kubectl get pods,svc
```

Expected service:
```
service/kube-ops-view   NodePort   <CLUSTER-IP>   <none>   80:32000/TCP
```

Access options:
- `kubectl port-forward svc/kube-ops-view 8080:80` → http://localhost:8080
- Or via NodePort: `http://<node-container-ip>:32000`
- If you configured `extraPortMappings`, `http://localhost:32000`

To stop/start later:
```powershell
kubectl scale deploy/kube-ops-view --replicas=0   # stop
kubectl scale deploy/kube-ops-view --replicas=1   # start
```

## D. Deploy API + UI with Kustomize

This repo includes individual manifests (`api-deploy.yaml`, `api-svc.yaml`, `streamlit-deploy.yaml`, `streamlit-svc.yaml`) and a `kustomization.yaml` that references them.

**Folder layout (example):**
```
deployment/kubernetes/
├─ api-deploy.yaml
├─ api-svc.yaml
├─ streamlit-deploy.yaml
├─ streamlit-svc.yaml
└─ kustomization.yaml
```

**Apply with Kustomize:**
```powershell
cd deployment/kubernetes
kubectl apply -k .
kubectl get all
```

**What the manifests do (defaults you can tune):**
- API (FastAPI) Deployment listens on **8005**; Service type **NodePort** @ **30100**.
- UI (Streamlit) Deployment listens on **8501**; Service type **NodePort** @ **30000**.
- The UI expects the API URL. Use one of:
  - In‑cluster DNS (if coded that way): `http://house-price-api:8005`
  - Or an explicit env var: set `API_URL` in the UI Deployment to `http://house-price-api:8005`
  - For local host access, port‑forward is simplest (see below).

**Port-forward for quick testing (works everywhere):**
```powershell
# API
kubectl port-forward svc/house-price-api 8005:8005
# UI
kubectl port-forward svc/house-price-ui  8501:8501
```
Open: http://localhost:8005/docs and http://localhost:8501

**Using NodePort instead (KIND caveat):**
- With default KIND networking, use the node container IP:
  - `docker inspect -f "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}" kind-control-plane`
  - API → `http://<ip>:30100`, UI → `http://<ip>:30000`
- If you added `extraPortMappings`, use `http://localhost:<nodePort>` directly.

## E. Generate manifests from kubectl (optional patterns)

You can “print but don’t create” YAML from commands, then keep them under version control:

```bash
kubectl create deployment house-price-api --image=sonvt8/house-price-api:latest --port=8005 \
  --dry-run=client -o yaml > api-deploy.yaml

kubectl create service nodeport house-price-api --tcp=8005 --node-port=30100 \
  --dry-run=client -o yaml > api-svc.yaml
```

```bash
kubectl create deployment house-price-ui --image=sonvt8/house-price-ui:k8s --port=8501\
  --dry-run=client -o yaml > streamlit-deploy.yaml

kubectl create service nodeport house-price-api --tcp=8501 --node-port=30100 \
  --dry-run=client -o yaml > streamlit-svc.yaml
```

- `--dry-run=client`: build the object locally without sending to the API server.
- `--dry-run=server`: ask the API server to validate/default, but **don’t persist**.
- After editing YAML, apply with `kubectl apply -k .` (Kustomize) or `kubectl apply -f <file>`.

## F. Image sourcing in KIND (important)

KIND’s nodes are **separate container runtimes**. If you want to use a **locally built** image (not pushed to a registry), load it into the cluster:

```bash
docker build -t house-price-api:0.1 .
kind load docker-image house-price-api:0.1
# Then reference image: house-price-api:0.1 in Deployment
# and prefer: imagePullPolicy: IfNotPresent
```

If you use Docker Hub images (e.g., `sonvt8/house-price-api:latest`), the nodes will pull them normally (add `imagePullSecrets` if private).

## G. Troubleshooting quick sheet

**Service not reachable**
- Ensure Service selector matches Pod labels:
  - `kubectl get pods --show-labels`
  - `kubectl get svc <name> -o yaml`
  - `kubectl get endpoints <name> -o wide`
- Prefer `kubectl port-forward` to bypass NodePort quirks in KIND.

**`ErrImagePull` / `ImagePullBackOff`**
- Tag and `kind load docker-image <name:tag>` for local images.
- Or push to registry and point Deployment to the full image name.

**`CrashLoopBackOff`**
- `kubectl logs deploy/<name> --tail=200`
- `kubectl logs pod/<name> --previous --tail=200`
- Verify command/entrypoint, env vars, ports bound to `0.0.0.0`, and required files/paths.

**Rollout / recreate**
```bash
kubectl rollout restart deploy/<name>
kubectl rollout status deploy/<name>
kubectl scale deploy/<name> --replicas=0 && kubectl scale deploy/<name> --replicas=1
```

**Clean up lab**
```bash
kubectl delete -k deployment/kubernetes
kind delete cluster
```

## I. Cleanup strategies — reclaim in levels (choose what you need)

Sometimes you only want to reclaim app resources, not the whole cluster. The commands below let you clean up **by layer**, so users can remove exactly what they want.

### Level 1 — App layer (UI & API only)
```bash
# Delete Deployments and Services for API & UI
kubectl delete deploy,svc house-price-api house-price-ui --ignore-not-found
```

### Level 2 — Observability/utility (kube-ops-view)
```bash
kubectl delete deploy kube-ops-view --ignore-not-found
kubectl delete svc kube-ops-view --ignore-not-found
```

### Level 3 — Kustomize stack (if deployed with kustomization.yaml)
```bash
# Run from the folder containing your kustomization.yaml
kubectl delete -k .
```

### Level 4 — Networking access (optional)
- Stop any `kubectl port-forward` processes (Ctrl+C in that terminal).
- If you created extra NodePorts/Ingresses manually, delete them as needed.

### Level 5 — KIND cluster
```bash
kind delete cluster
```

### Level 6 — Docker housekeeping on host (optional)
```bash
# Remove images you no longer need (example)
docker rmi sonvt8/house-price-api:latest sonvt8/house-price-ui:latest || true
# Reclaim dangling resources (be cautious)
docker system prune -f
```

## H. What users should see working

- **kube-ops-view** UI reachable (via port-forward or NodePort) and showing 3 nodes.
- **FastAPI** reachable at `/health` and `/docs`.
- **Streamlit** reachable and able to call the API (check logs for the resolved `API_URL`).

> If something fails, run `kubectl get events --sort-by=.lastTimestamp -A` and `kubectl describe` on the failing resource first, then check container logs.

---

# Development Workflow — Code Quality & Pre-commit Rules

This section outlines the mandatory code quality standards and pre-commit workflow that must be followed when contributing new code to this MLOps project.

## Prerequisites for Development

Before contributing code, ensure you have the following tools installed:

```bash
# Install development dependencies
pip install ruff black pre-commit

# Install pre-commit hooks
pre-commit install
```

## Code Quality Standards

This project enforces strict code quality standards through automated tools:

### 1. **Code Formatting (Black)**
- **Tool**: Black formatter
- **Standard**: PEP 8 with 99-character line length
- **Auto-fix**: Yes (automatic on commit)

### 2. **Code Linting (Ruff)**
- **Tool**: Ruff linter
- **Rules**: 15+ quality rules including:
  - **E, W**: PEP 8 style guidelines
  - **F**: Logic errors (unused imports, variables)
  - **B**: Bug detection (flake8-bugbear)
  - **S**: Security issues (bandit)
  - **D**: Documentation standards
  - **N**: Naming conventions
  - **T**: Type checking hints
  - **Q**: Code complexity (McCabe)
  - **R**: Refactoring suggestions
  - **PIE**: Performance optimization
  - **SIM**: Code simplification
  - **TCH**: Type checking improvements
- **Auto-fix**: Yes (90%+ of issues)

### 3. **File Quality Checks**
- **Trailing whitespace**: Automatically removed
- **End of file**: Ensures proper newline endings
- **JSON syntax**: Validates JSON files
- **Large files**: Warns about files >1MB
- **Merge conflicts**: Detects unresolved conflicts
- **Line endings**: Fixes mixed CRLF/LF issues

## Mandatory Pre-commit Workflow

### Step 1: Code Development
```bash
# 1. Create feature branch
git checkout -b feature/your-feature-name

# 2. Make your changes
# ... edit files ...

# 3. Test your changes locally
python -m pytest tests/  # if tests exist
```

### Step 2: Pre-commit Validation
```bash
# 4. Run pre-commit checks manually (recommended)
pre-commit run --all-files

# This will automatically:
# - Format code with Black
# - Lint and fix with Ruff
# - Remove trailing whitespace
# - Fix end-of-file issues
# - Validate JSON syntax
# - Check for merge conflicts
# - Fix line ending issues
```

### Step 3: Commit Process
```bash
# 5. Stage your changes
git add .

# 6. Commit (pre-commit hooks run automatically)
git commit -m "feat: add new feature"

# If pre-commit made changes, you'll see:
# "Files were modified by this hook"
# Re-stage and commit again:
git add .
git commit -m "feat: add new feature"
```

### Step 4: Push to Repository
```bash
# 7. Push to remote
git push origin feature/your-feature-name

# 8. Create Pull Request on GitHub
# The CI/CD pipeline will validate your changes
```

## Code Quality Rules by File Type

### **Python Files** (`*.py`)
- ✅ **Black formatting** (99-character lines)
- ✅ **Ruff linting** (15+ quality rules)
- ✅ **Security checks** (bandit)
- ✅ **Complexity limits** (McCabe ≤10)
- ✅ **Type hints** (encouraged)
- ✅ **Docstrings** (for public functions)

### **Jupyter Notebooks** (`*.ipynb`)
- ✅ **nbQA-Black** formatting
- ✅ **nbQA-Ruff** linting
- ✅ **Relaxed rules** for notebooks (print statements, unused imports allowed)

### **Configuration Files**
- ✅ **YAML validation** (disabled due to encoding issues)
- ✅ **JSON validation**
- ✅ **Trailing whitespace removal**
- ✅ **End-of-file fixes**

### **Docker Files**
- ✅ **Hadolint** (Dockerfile linting)
- ✅ **File format checks**

## Quality Gates

Your code **MUST** pass all quality gates before being merged:

### **Local Quality Gates**
```bash
# All checks must pass
pre-commit run --all-files
# Expected output: All checks passed ✅
```

### **CI/CD Quality Gates**
- ✅ **GitHub Actions** validation
- ✅ **Docker build** success
- ✅ **API health checks**
- ✅ **End-to-end pipeline** validation

## Common Issues & Solutions

### **Issue**: Pre-commit fails with formatting errors
```bash
# Solution: Let pre-commit fix automatically
pre-commit run --all-files
git add .
git commit -m "fix: auto-format code"
```

### **Issue**: Ruff linting errors
```bash
# Solution: Auto-fix with Ruff
ruff check --fix .
git add .
git commit -m "fix: resolve linting issues"
```

### **Issue**: Large file warnings
```bash
# Solution: Add to .gitignore or use Git LFS
echo "large_file.csv" >> .gitignore
git rm --cached large_file.csv
```

### **Issue**: Merge conflict markers
```bash
# Solution: Resolve conflicts manually
# Remove <<<<<<< ======= >>>>>>> markers
# Then commit the resolved version
```

## Bypassing Pre-commit (Emergency Only)

⚠️ **WARNING**: Only use in emergencies and fix immediately after:

```bash
# Skip pre-commit for this commit only
git commit --no-verify -m "emergency: critical fix"

# Immediately fix and recommit
pre-commit run --all-files
git add .
git commit -m "fix: apply code quality standards"
```

## Team Responsibilities

### **Developers Must**:
1. ✅ Run `pre-commit run --all-files` before committing
2. ✅ Fix all auto-fixable issues
3. ✅ Address security warnings
4. ✅ Keep functions under complexity limit (McCabe ≤10)
5. ✅ Write meaningful commit messages

### **Code Reviewers Must**:
1. ✅ Verify all quality gates passed
2. ✅ Check for security issues
3. ✅ Ensure proper documentation
4. ✅ Validate test coverage

### **CI/CD Pipeline**:
1. ✅ Automatically validates all changes
2. ✅ Blocks merge if quality gates fail
3. ✅ Provides detailed error reports
4. ✅ Enforces consistent code standards

## Benefits of This Workflow

- 🚀 **Faster development**: Auto-fix eliminates manual formatting
- 🛡️ **Higher quality**: Catches bugs and security issues early
- 📏 **Consistent style**: All code follows same standards
- 🔍 **Better reviews**: Focus on logic, not formatting
- 🚫 **Fewer bugs**: Automated quality checks prevent issues
- 📚 **Self-documenting**: Clear standards for new team members

---

**Remember**: These quality standards are not optional. They ensure the reliability, maintainability, and security of our MLOps pipeline. Every commit must pass all quality gates before being merged into the main branch.
