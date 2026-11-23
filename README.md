# RoadGuard AI
### Intelligent Road Infrastructure Monitoring System

RoadGuard AI is an image-based road damage classification and monitoring system powered by deep learning.

It demonstrates an end-to-end machine learning workflow, including data preparation, image preprocessing, model training, evaluation, API development, dashboard monitoring, candidate retraining, Docker deployment, cloud deployment, and load testing.

## 1. Project Objectives

- Classify road images into selected damage categories.
- Provide image predictions through a Python API.
- Display predictions through a web dashboard.
- Monitor API health and response times.
- Support bulk image uploads and candidate retraining.
- Evaluate candidate models before promotion.
- Test API performance under concurrent requests.
- Deploy the application to AWS EC2.

## 2. Road Damage Categories

| Code | Description |
|---|---|
| D00 | Longitudinal cracks |
| D10 | Transverse cracks |
| D20 | Alligator cracks |
| D40 | Potholes |

This is a multi-label classification task. An image can contain multiple selected damage classes.

An image with none of these four labels is not necessarily damage-free; it may contain other damage categories.

## 3. Dataset

The project uses the India and Czech subsets of RDD2022.

**Dataset source:**

[Road Damage Dataset 2022 (RDD2022)](https://figshare.com/articles/dataset/RDD2022_-_The_multi-national_Road_Damage_Dataset_released_through_CRDDC_2022/21431547)

Dataset preparation:

| Dataset component | Images |
|---|---:|
| India images | 7,706 |
| India training | 5,394 |
| India validation | 1,156 |
| India testing | 1,156 |
| Czech additional training images | 2,829 |
| Combined training images | 8,223 |

The original India validation and test sets are retained for evaluation.

Raw images are excluded from this repository because of their size. Download the dataset separately and arrange it according to the paths expected by the preprocessing scripts.

## 4. Model Architecture

The model uses MobileNetV2 with ImageNet pretrained weights.

Architecture:

1. MobileNetV2 feature extractor
2. Global Average Pooling
3. Dropout
4. Dense output layer with four sigmoid units

Configuration:

| Parameter | Value |
|---|---|
| Architecture | MobileNetV2 |
| Input size | 160 × 160 |
| Output classes | 4 |
| Optimizer | Adam |
| Loss | Binary cross-entropy |
| Task | Multi-label classification |

Class-specific thresholds convert probabilities into predictions.

## 5. Model Evaluation

Metrics include macro precision, macro recall, macro F1, micro F1, and macro ROC-AUC.

### Previous versus candidate model

| Metric | Previous | Candidate |
|---|---:|---:|
| Macro F1 | 0.3572 | 0.3816 |
| Macro precision | 0.3161 | 0.3377 |
| Macro recall | 0.4279 | 0.4671 |
| Micro F1 | 0.4676 | 0.4868 |
| Macro ROC-AUC | 0.7516 | 0.7552 |

The candidate improved the reported aggregate metrics and passed the configured promotion checks.

D10 has relatively few positive test examples, so its evaluation is less reliable.

The test split was used in multiple development evaluations. These results should be interpreted as development results, not as a pristine independent final evaluation.

## 6. Project Structure

```text
roadguard-ai/
├── api/
├── dashboard/
├── data/
│   ├── metadata/
│   └── processed/
├── locust/
├── models/
│   ├── roadguard_best.keras
│   ├── roadguard_final.keras
│   └── thresholds.json
├── nginx/
│   └── nginx.conf
├── notebooks/
│   └── roadguard_model.ipynb
├── src/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-docker.txt
└── README.md
```

## 7. Local Installation and Setup

### Prerequisites

- Python 3.13
- Docker
- Docker Compose
- Git

### Clone the repository

```bash
git clone https://github.com/codestantceasar/RoadGuard-AI.git
cd RoadGuard-AI
```

### Install Python dependencies

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Run the application using Docker

Build and start the API, Nginx, and dashboard:

```bash
docker compose up -d --build api nginx dashboard
```

Check running containers:

```bash
docker compose ps
```

View application logs:

```bash
docker compose logs -f
```

### Local application URLs

| Service | URL |
|---|---|
| Dashboard | http://localhost:8501 |
| API | http://localhost:8000 |
| API documentation | http://localhost:8000/docs |
| Health check | http://localhost:8000/health |

## 8. API Endpoints

The FastAPI application provides the following endpoints:

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API information |
| GET | `/health` | API health and model status |
| POST | `/predict` | Upload an image and predict road damage |

The prediction endpoint accepts JPEG and PNG images up to 10 MB.

Example health check:

```bash
curl http://localhost:8000/health
```

Interactive API documentation is available at:

`http://localhost:8000/docs`

## 9. Dashboard Features

The Streamlit dashboard provides:

- API health monitoring.
- Response-time monitoring.
- Road damage dataset analytics.
- Road damage class distribution.
- Damage class co-occurrence analysis.
- Single-image prediction.
- Bulk image upload and prediction.
- Candidate model retraining.
- Candidate evaluation and promotion workflow.

The dashboard provides a user-friendly interface for interacting with the road damage classification system.

## 10. Candidate Retraining and Model Promotion

RoadGuard AI supports an experimental candidate retraining workflow.

The workflow includes:

1. Uploading training data.
2. Preparing the training dataset.
3. Training a candidate model.
4. Evaluating candidate performance.
5. Comparing candidate metrics with the existing model.
6. Applying configured promotion checks.
7. Promoting an accepted candidate model.

This workflow helps demonstrate model lifecycle management and controlled model updates.

## 11. Load Testing with Locust

Locust is used to evaluate API performance under concurrent requests.

The project includes a Locust test configuration for API endpoints.

Start the optional Locust service:

```bash
docker compose --profile loadtest up -d locust
```

Access the Locust interface:

`http://localhost:8089`

### Local Load Test Results

The recorded local load test used:

| Configuration | Value |
|---|---:|
| Concurrent users | 10 |
| Spawn rate | 2 users/second |
| Duration | Approximately 2 minutes |
| Total requests | 570 |
| Failed requests | 0 |
| Average response time | 91.96 ms |
| Median response time | 100 ms |
| 95th percentile | 140 ms |
| 99th percentile | 230 ms |
| Requests per second | 5.1 |

### Endpoint performance

| Endpoint | Requests | Failures | Average response |
|---|---:|---:|---:|
| GET `/health` | 113 | 0 | 10.91 ms |
| POST `/predict` | 457 | 0 | 112 ms |

These measurements represent the recorded local Docker load test. They should not be interpreted as AWS cloud load-testing results.

## 12. AWS Cloud Deployment

RoadGuard AI was successfully deployed to an Amazon EC2 instance running Ubuntu Server 24.04 LTS.

### Cloud Infrastructure

| Component | Configuration |
|---|---|
| Cloud provider | Amazon Web Services |
| Service | Amazon EC2 |
| Operating system | Ubuntu Server 24.04 LTS |
| AWS region | Stockholm (`eu-north-1`) |
| Instance type | m7i-flex.large |
| Container runtime | Docker |
| Orchestration | Docker Compose |
| Reverse proxy | Nginx |

### Live Deployment

**Dashboard:**

http://13.48.29.8:8501

**API documentation:**

http://13.48.29.8:8000/docs

**API health endpoint:**

http://13.48.29.8:8000/health

Note: Dashboard access is restricted through the AWS security group to an authorized IP address. The EC2 public IP may change if the instance is stopped and restarted.

### Deployment Architecture

The cloud deployment uses three primary services:

1. **FastAPI:** Handles image uploads and road damage predictions.
2. **Nginx:** Routes incoming API requests to the FastAPI service.
3. **Streamlit:** Provides the interactive monitoring and prediction dashboard.

Docker Compose manages the application containers.

### Cloud Deployment Verification

The following deployment checks were completed successfully:

| Test | Result |
|---|---|
| EC2 instance | Running |
| Docker installation | Successful |
| API container | Healthy |
| Nginx container | Healthy |
| Dashboard container | Running |
| API health check | Online |
| Dashboard access | Successful |
| API documentation | Accessible |
| Image upload | Successful |
| Model prediction | Successful |

### Cloud Prediction Test

A road image named `India_000072.jpg` was uploaded through the deployed Streamlit dashboard.

The model successfully processed the image and identified possible pothole damage.

Recorded test results:

| Measurement | Result |
|---|---:|
| Prediction response time | 638.03 ms |
| API health-check response time | 2.99 ms |
| Input image size | 71.3 KB |
| Model architecture | MobileNetV2 |
| Input dimensions | 160 × 160 |
| Prediction classes | 4 |

These measurements represent an individual successful cloud prediction test, not a comprehensive performance benchmark.

## 13. Technologies Used

| Category | Technologies |
|---|---|
| Programming language | Python |
| Deep learning | TensorFlow, Keras |
| Model architecture | MobileNetV2 |
| Image processing | Pillow, NumPy |
| API development | FastAPI |
| Dashboard | Streamlit |
| Reverse proxy | Nginx |
| Containerization | Docker |
| Container orchestration | Docker Compose |
| Load testing | Locust |
| Cloud infrastructure | AWS EC2 |
| Version control | Git, GitHub |

## 14. Limitations

- The model is trained on selected subsets of RDD2022.
- Some damage classes are underrepresented.
- D10 classification performance is relatively weak.
- Model predictions are experimental and may produce false positives or false negatives.
- Dataset distribution may differ from real-world road conditions.
- The evaluation results are development results and should not be treated as independent final validation.
- Cloud prediction response times may vary depending on infrastructure and network conditions.

RoadGuard AI is an experimental decision-support system. Predictions should be reviewed by qualified road inspectors before infrastructure decisions are made.

## 15. Future Improvements

- Improve classification performance for underrepresented damage classes.
- Expand training data with more geographically diverse road images.
- Introduce stronger model monitoring and drift detection.
- Implement automated model versioning and experiment tracking.
- Add authentication and HTTPS for production deployment.
- Improve cloud infrastructure scalability.
- Establish an independent evaluation dataset.
- Explore automated retraining and deployment pipelines.

## 16. Author and Repository

**GitHub Repository:**

https://github.com/codestantceasar/RoadGuard-AI

**Project:** RoadGuard AI – Intelligent Road Infrastructure Monitoring System

---

**RoadGuard AI demonstrates an end-to-end machine learning operations workflow, from dataset preparation and model evaluation to containerization, cloud deployment, monitoring, prediction, and performance testing.**
