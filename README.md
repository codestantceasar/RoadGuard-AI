# RoadGuard AI
## Project Demonstration Video

Watch the RoadGuard AI project demonstration:

[**Watch the RoadGuard AI Presentation Video**](https://drive.google.com/file/d/1u_lPZ_R61n9_FrmUmB1LnXKhVVp83WKu/view?usp=drive_link)

The video demonstrates the RoadGuard AI dashboard, road damage prediction, model retraining workflow, evaluation results, and AWS deployment.

### Intelligent Road Infrastructure Monitoring System

RoadGuard AI is an image-based road damage classification and monitoring system powered by deep learning. It demonstrates an end-to-end Machine Learning Operations (MLOps) workflow, covering data preparation, model training, evaluation, API development, dashboard monitoring, candidate retraining, Docker deployment, AWS cloud deployment, and load testing.

The system analyzes road images and predicts the presence of four selected road damage categories using a multi-label classification model.

---

## 1. Project Objectives

RoadGuard AI aims to:

- Classify road images into selected damage categories.
- Provide image predictions through a Python REST API.
- Display predictions through an interactive web dashboard.
- Monitor API health and response times.
- Support bulk image uploads and labelled dataset collection.
- Retrain a separate candidate model using newly collected data.
- Evaluate candidate models before considering promotion.
- Test API performance under concurrent requests.
- Deploy the application to AWS EC2 using Docker Compose.

## 2. Road Damage Categories

The project focuses on four road damage categories from the Road Damage Dataset 2022 (RDD2022).

| Class | Description |
|---|---|
| D00 | Longitudinal cracks |
| D10 | Transverse cracks |
| D20 | Alligator cracks |
| D40 | Potholes |

This is a **multi-label classification problem**. An image may contain more than one damage category.

An image without any of these four selected labels is not necessarily damage-free; it may contain other road damage categories outside the scope of this model.

## 3. Dataset

The primary dataset is based on the India and Czech subsets of the Road Damage Dataset 2022 (RDD2022).

### Dataset composition

| Dataset component | Images |
|---|---:|
| India images | 7,706 |
| India training images | 5,394 |
| India validation images | 1,156 |
| India testing images | 1,156 |
| Czech additional training images | 2,829 |
| Combined original training dataset | 8,223 |

An additional RoadDamageVision dataset was incorporated into the experimental retraining workflow.

### Dataset preparation

The data preparation process includes:

1. Loading image annotation metadata.
2. Checking image availability.
3. Removing records with missing image files.
4. Extracting the selected damage categories.
5. Converting damage labels into binary multi-label targets.
6. Preparing training, validation, and testing CSV files.
7. Validating image paths and labels before training.

The original India validation and testing datasets are retained separately from the original training dataset.

Raw image datasets are excluded from this repository because of their size and licensing considerations. Users must obtain the datasets from their respective sources and arrange them according to the expected directory structure.

### External dataset

RoadDamageVision was incorporated into the retraining workflow.

The external dataset is stored locally and on the deployment server rather than committed to GitHub.

The RoadDamageVision dataset is distributed under CC BY 4.0. Users must follow its attribution requirements when obtaining or redistributing the dataset.

## 4. Model Architecture

RoadGuard AI uses MobileNetV2 with ImageNet pretrained weights as its feature extraction backbone.

### Architecture

- MobileNetV2 feature extractor
- Global Average Pooling
- Dropout layer
- Dense output layer with four sigmoid units

### Model configuration

| Parameter | Value |
|---|---|
| Architecture | MobileNetV2 |
| Input dimensions | 160 × 160 |
| Input channels | 3 (RGB) |
| Output classes | 4 |
| Optimizer | Adam |
| Loss function | Binary cross-entropy |
| Task | Multi-label classification |
| Output activation | Sigmoid |
| Initial learning rate | 0.0001 |

The model includes an internal rescaling layer that converts image pixel values from the 0–255 range to the -1 to 1 range expected by MobileNetV2.

Class-specific thresholds are used to convert predicted probabilities into final damage classifications.

## 5. Model Evaluation

RoadGuard AI evaluates model performance using:

- Macro precision
- Macro recall
- Macro F1-score
- Micro F1-score
- Macro ROC-AUC

### Production versus candidate model

The latest candidate retraining experiment compared the existing production model against a newly trained candidate model.

| Metric | Production model | Candidate model |
|---|---:|---:|
| Macro precision | 0.3377 | 0.3194 |
| Macro recall | 0.4671 | 0.4287 |
| Macro F1 | 0.3816 | 0.3500 |
| Micro F1 | 0.4868 | 0.4393 |
| Macro ROC-AUC | 0.7552 | 0.7364 |

### Evaluation outcome

The candidate model did not meet the configured promotion criteria.

Its macro F1-score decreased from 0.3816 to 0.3500 compared with the existing production model.

The candidate was therefore rejected, and the existing production model remained unchanged.

This demonstrates the use of an evaluation-based model promotion process rather than automatically replacing a working model after retraining.

### Class-level observations

D10 classification remains a significant weakness, with relatively low precision and F1 performance.

This indicates that further improvements in data representation, class balance, and model training are necessary.

### Evaluation limitations

The original test split has been used in development comparisons. Consequently, the reported metrics should be interpreted as development evaluation results rather than a completely independent final assessment.

An additional independent evaluation using the external RoadDamageVision test split is a potential next step, subject to verification of its image paths, labels, and suitability.

## 6. Project Structure

```text
RoadGuard-AI/
│
├── api/
│   └── FastAPI application files
│
├── dashboard/
│   └── Streamlit dashboard files
│
├── data/
│   ├── metadata/
│   ├── processed/
│   └── uploads/
│
├── locust/
│   └── test_data/
│
├── models/
│   ├── roadguard_best.keras
│   ├── roadguard_final.keras
│   └── thresholds.json
│
├── nginx/
│   └── nginx.conf
│
├── notebooks/
│   └── roadguard_model.ipynb
│
├── outputs/
│   └── cloud-loadtest/
│
├── src/
│   ├── model.py
│   ├── preprocessing.py
│   ├── image_loader.py
│   ├── prediction.py
│   ├── retraining.py
│   └── evaluation.py
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-docker.txt
├── create_notebook.py
├── .dockerignore
├── .gitignore
└── README.md
```

Note: The structure above highlights the principal project components. Additional source files and generated artifacts may also be present.

Raw image datasets and large external datasets are not included in the GitHub repository.

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

### Create a virtual environment

Linux/macOS:

```bash
python3 -m venv .venv

source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv

.venv\Scripts\Activate.ps1
```

### Install dependencies

```bash
pip install --upgrade pip

pip install -r requirements.txt
```

### Run the application using Docker Compose

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

Stop the application:

```bash
docker compose down
```

### Local application URLs

| Service | URL |
|---|---|
| Dashboard | http://localhost:8501 |
| API | http://localhost:8000 |
| API documentation | http://localhost:8000/docs |
| Health check | http://localhost:8000/health |

**Dataset requirement:** The raw and processed datasets must be available in their expected locations before running training or retraining workflows.

## 8. API Endpoints

The FastAPI application provides REST endpoints for interacting with the prediction model.

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

Example response:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

Interactive API documentation is available at:

http://localhost:8000/docs

## 9. Dashboard Features

The Streamlit dashboard provides an interactive interface for monitoring and using RoadGuard AI.

### System monitoring

- API availability monitoring.
- API health checks.
- Response-time monitoring.

### Dataset analytics

- Road damage dataset statistics.
- Damage class distribution.
- Multiple damage class analysis.
- Class co-occurrence visualizations.

### Prediction

- Single-image upload.
- Image-based damage classification.
- Multi-label prediction results.
- Class-specific probability thresholds.

### Bulk upload and retraining

- Bulk image uploads.
- Label validation.
- Dataset storage.
- Candidate model retraining.
- Candidate evaluation.
- Controlled model promotion checks.

## 10. Candidate Retraining and Model Promotion

RoadGuard AI implements an experimental candidate retraining workflow.

The objective is to allow new labelled images to contribute to model improvement without automatically replacing the existing production model.

### Retraining workflow

1. Upload labelled road images.
2. Validate uploaded image files and labels.
3. Store upload records in SQLite.
4. Combine the original training data, external training data, and validated uploads.
5. Prepare the TensorFlow training dataset.
6. Load the existing production model.
7. Train a separate candidate model.
8. Save the candidate model and training history.
9. Evaluate the candidate against the existing production model.
10. Apply configured promotion criteria.

### Candidate model protection

The production model is stored separately from the candidate model.

| Model | Location |
|---|---|
| Production model | `models/roadguard_best.keras` |
| Candidate model | `outputs/retraining/roadguard_candidate.keras` |

The candidate is not automatically promoted.

In the latest experiment, the candidate failed the macro F1 promotion criterion and was rejected. The production model remained unchanged.

This workflow demonstrates controlled model retraining and evaluation-based model lifecycle management.

## 11. Load Testing with Locust

Locust was used to evaluate API performance under concurrent requests.

The tests covered:

- `GET /health`: Checks API availability and model status.
- `POST /predict`: Uploads an image and checks the prediction response.

### Local load test

Configuration:

| Parameter | Result |
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

Endpoint performance:

| Endpoint | Requests | Failures | Average response |
|---|---:|---:|---:|
| GET /health | 113 | 0 | 10.91 ms |
| POST /predict | 457 | 0 | 112 ms |

### AWS cloud load test

The deployed AWS application was also tested using Locust.

Configuration and results:

| Parameter | Result |
|---|---:|
| Concurrent users | 10 |
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

AWS endpoint performance:

| Endpoint | Requests | Failures | Average response |
|---|---:|---:|---:|
| GET /health | 105 | 0 | 9.29 ms |
| POST /predict | 474 | 0 | 72.63 ms |
| **Total** | **579** | **0** | **61.15 ms** |

The AWS test completed all recorded requests without failures or Locust exceptions.

These results represent one controlled cloud load-test run and should not be interpreted as the maximum capacity of the deployment.

### Load test reports

The repository contains the following recorded cloud load-test reports:

```text
outputs/cloud-loadtest/
├── aws_locust_exceptions.csv
├── aws_locust_failures.csv
├── aws_locust_report.html
├── aws_locust_stats.csv
└── aws_locust_stats_history.csv
```

These files contain request statistics, failure information, response-time history, and an HTML report.

### Run Locust locally

Start the optional load-testing service:

```bash
docker compose --profile loadtest up -d locust
```

Access the Locust interface:

http://localhost:8089

## 12. AWS Cloud Deployment

RoadGuard AI was deployed to an Amazon EC2 instance running Ubuntu Server 24.04 LTS.

### Cloud infrastructure

| Component | Configuration |
|---|---|
| Cloud provider | Amazon Web Services |
| Service | Amazon EC2 |
| Operating system | Ubuntu Server 24.04 LTS |
| AWS region | Stockholm (eu-north-1) |
| Availability zone | eu-north-1a |
| Instance type | m7i-flex.large |
| Container runtime | Docker |
| Orchestration | Docker Compose |
| Reverse proxy | Nginx |

### Live deployment

**Dashboard**

http://13.48.29.8:8501

**API documentation**

http://13.48.29.8:8000/docs

**API health endpoint**

http://13.48.29.8:8000/health

Note: These URLs use the EC2 public IP recorded during deployment. The address may change if the instance is stopped and restarted. Dashboard access is restricted through the AWS security group to an authorized IP address.

### Deployment architecture

The cloud deployment uses three primary services:

- **FastAPI:** Handles image uploads and road damage predictions.
- **Nginx:** Acts as a reverse proxy for API traffic.
- **Streamlit:** Provides the interactive monitoring and prediction dashboard.

Docker Compose manages the application containers.

### Cloud deployment verification

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

### Cloud prediction test

A road image named `India_000072.jpg` was uploaded through the deployed Streamlit dashboard.

The model successfully processed the image and identified possible pothole damage.

Recorded measurements:

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
| Data processing | Pandas |
| API development | FastAPI |
| Dashboard | Streamlit |
| Database | SQLite |
| Reverse proxy | Nginx |
| Containerization | Docker |
| Container orchestration | Docker Compose |
| Load testing | Locust |
| Cloud infrastructure | AWS EC2 |
| Version control | Git, GitHub |
| Notebook environment | Jupyter Notebook |

## 14. Limitations

RoadGuard AI is an experimental road damage monitoring system and has several limitations:

- The model is trained on selected subsets of RDD2022 and additional external data.
- Some damage classes are underrepresented.
- D10 classification performance remains relatively weak.
- Predictions may produce false positives or false negatives.
- Dataset distribution may differ from real-world road conditions.
- The original test split has been used in development comparisons.
- The latest candidate model did not outperform the existing production model.
- Cloud response times may vary depending on infrastructure and network conditions.
- The current cloud deployment does not implement full production-grade authentication or security controls.

RoadGuard AI should be treated as an experimental decision-support system. Its predictions should be reviewed by qualified road inspectors before infrastructure decisions are made.

## 15. Future Improvements

Potential improvements include:

- Improve classification performance for underrepresented damage classes.
- Expand training data with geographically diverse road images.
- Perform independent evaluation using an untouched external test dataset.
- Introduce stronger model monitoring and data drift detection.
- Implement automated model versioning and experiment tracking.
- Add authentication and HTTPS for production deployment.
- Improve cloud infrastructure scalability.
- Introduce more robust model promotion and rollback mechanisms.
- Explore automated retraining and deployment pipelines.

## 16. Author and Repository

**GitHub Repository:**

https://github.com/codestantceasar/RoadGuard-AI

**Project:** RoadGuard AI – Intelligent Road Infrastructure Monitoring System

RoadGuard AI demonstrates an end-to-end machine learning and MLOps workflow, from dataset preparation and model evaluation to API development, dashboard monitoring, containerization, cloud deployment, candidate retraining, and performance testing.

---

*RoadGuard AI — Applying Machine Learning to Intelligent Road Infrastructure Monitoring.*