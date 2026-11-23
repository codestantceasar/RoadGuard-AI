# RoadGuard AI
### Intelligent Road Infrastructure Monitoring System

RoadGuard AI is an image-based road damage classification and monitoring system powered by deep learning.

It demonstrates an end-to-end machine learning workflow, including data preparation, image preprocessing, model training, evaluation, API development, dashboard monitoring, candidate retraining, Docker deployment and load testing.

## 1. Project Objectives

- Classify road images into selected damage categories.
- Provide image predictions through a Python API.
- Display predictions through a web dashboard.
- Monitor API health and response times.
- Support bulk image uploads and candidate retraining.
- Evaluate candidate models before promotion.
- Test API performance under concurrent requests.

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

Dataset source:

https://figshare.com/articles/dataset/RDD2022_-_The_multi-national_Road_Damage_Dataset_released_through_CRDDC_2022/21431547

Dataset preparation:
- India images: 7,706
- India training: 5,394
- India validation: 1,156
- India testing: 1,156
- Czech additional training images: 2,829
- Combined training images: 8,223

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
- Input size: 160 × 160
- Output classes: 4
- Optimizer: Adam
- Loss: Binary cross-entropy
- Task: Multi-label classification

Class-specific thresholds convert probabilities into predictions.

## 5. Model Evaluation

Metrics include macro precision, macro recall, macro F1, micro F1 and macro ROC-AUC.

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
├── nginx/
├── notebooks/
├── src/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── requirements-docker.txt