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

**Dataset source:** [Road Damage Dataset 2022 (RDD2022)](https://figshare.com/articles/dataset/RDD2022_-_The_multi-national_Road_Damage_Dataset_released_through_CRDDC_2022/21431547)

### Dataset preparation

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

### Architecture

1. MobileNetV2 feature extractor
2. Global Average Pooling
3. Dropout
4. Dense output layer with four sigmoid units

### Configuration

| Parameter | Value |
|---|---|
| Architecture | MobileNetV2 |
| Input size | 160 × 160 |
| Output classes | 4 |
| Optimizer | Adam |
| Loss | Binary cross-entropy |
| Task | Multi-label classification |

Class-specific thresholds convert predicted probabilities into final class predictions.

## 5. Model Evaluation

Evaluation metrics include macro precision, macro recall, macro F1, micro F1, and macro ROC-AUC.

### Previous versus candidate model

| Metric | Previous | Candidate |
|---|---:|---:|
| Macro F1 | 0.3572 | 0.3816 |
| Macro precision | 0.3161 | 0.3377 |
| Macro recall | 0.4279 | 0.4671 |
| Micro F1 | 0.4676 | 0.4868 |
| Macro ROC-AUC | 0.7516 | 0.7552 |

The candidate improved all reported aggregate metrics and passed the configured promotion checks.

D10 has relatively few positive test examples, so its evaluation is less reliable.

**Evaluation limitation:** The test split was used in multiple development evaluations. These results should therefore be interpreted as development results, not as a pristine independent final evaluation.

## 6. Project Structure

```text
RoadGuard-AI/
├── api/
├── dashboard/
├── data/
│   ├── metadata/
│   ├── processed/
│   └── uploads/
├── locust/
│   └── test_data/
├── models/
│   ├── roadguard_best.keras
│   ├── roadguard_final.keras
│   └── thresholds.json
├── nginx/
│   └── nginx.conf
├── notebooks/
│   └── roadguard_model.ipynb
├── outputs/
│   └── cloud-loadtest/
├── src/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-docker.txt
└── README.md
```

Raw dataset images are not included in the repository.

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

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

Activate it on Windows PowerShell:

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

### Example health check

```bash
curl http://localhost:8000/health
```

Interactive API documentation:

http://localhost:8000/docs

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

This workflow demonstrates model lifecycle management and controlled model updates.

## 11. Load Testing with Locust

Locust is used to evaluate API performance under concurrent requests.

The project includes a Locust test configuration that tests:

- `GET /health`: Checks API availability and model status.
- `POST /predict`: Uploads an image and checks the prediction response.

### Run Locust locally

Start the optional Locust service:

```bash
docker compose --profile loadtest up -d locust
```

Access the Locust interface:

http://localhost:8089

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

### Local endpoint performance

| Endpoint | Requests | Failures | Average response |
|---|---:|---:|---:|
| GET `/health` | 113 | 0 | 10.91 ms |
| POST `/predict` | 457 | 0 | 112 ms |

These measurements represent the recorded local Docker load test and are separate from the AWS cloud results.

### AWS Cloud Load Test Results

The deployed AWS application was also tested using Locust.

**Test configuration:**

| Configuration | Value |
|---|---:|
| Virtual users | 10 |
| Spawn rate | 2 users/second |
| Duration | 2 minutes |
| Total requests | 579 |
| Failed requests | 0 |
| Average response time | 61.15 ms |
| Median response time | 63 ms |
| 90th percentile | 97 ms |
| 95th percentile | 120 ms |
| 99th percentile | 160 ms |
| Maximum response time | 290 ms |
| Throughput | 4.86 requests/second |

### AWS endpoint performance

| Endpoint | Requests | Failures | Average response |
|---|---:|---:|---:|
| GET `/health` | 105 | 0 | 9.29 ms |
| POST `/predict` | 474 | 0 | 72.63 ms |
| **Total** | **579** | **0** | **61.15 ms** |

The AWS test completed all recorded requests without failures or Locust exceptions.

These measurements represent one controlled cloud load-test run. They should not be interpreted as the maximum capacity of the deployment.

### Load Test Reports

Locust generated the following report files:

```text
outputs/cloud-loadtest/
├── aws_locust_exceptions.csv
├── aws_locust_failures.csv
├── aws_locust_report.html
├── aws_locust_stats.csv
└── aws_locust_stats_history.csv
```

These files contain detailed request statistics, failure information, response-time history, and an HTML report.

## 12. AWS Cloud Deployment

RoadGuard AI was deployed to an Amazon EC2 instance running Ubuntu Server 24.04 LTS.

### Cloud Infrastructure

| Component | Configuration |
|---|---|
| Cloud provider | Amazon Web Services |
| Service | Amazon EC2 |
| Operating system | Ubuntu Server 24.04 LTS |
| AWS region | Stockholm (`eu-north-1`) |
| Instance type | `m7i-flex.large` |
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

**Note:** These URLs use the EC2 public IP recorded during deployment. The IP may change if the instance is stopped and restarted. Dashboard access is restricted through the AWS security group to an authorized IP address.

### Deployment Architecture

The cloud deployment uses three primary services:

1. **FastAPI:** Handles image uploads and road damage predictions.
2. **Nginx:** Routes incoming API requests to the FastAPI service.
3. **Streamlit:** Provides the interactive monitoring and prediction dashboard.

Docker Compose manages the application containers.

### Cloud Deployment Verification

The following checks were completed successfully:

| Test | Result |
|---|---|
| EC2 instance | Running |
| Docker installation | Successful |
| API container | Healthy |
| Nginx container | Healthy |
| Dashboard container | Running |
| API health check | Successful |
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
- Evaluation results are development results and should not be treated as independent final validation.
- Cloud prediction response times may vary depending on infrastructure and network conditions.
- The current cloud deployment does not implement full production-grade authentication or security controls.

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