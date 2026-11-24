import sys
import os
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import requests
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


# ==================================================
# CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="RoadGuard AI",
    page_icon="🛣️",
    layout="wide"
)

API_URL = os.getenv("API_URL", "http://localhost:8000")

UPLOAD_IMAGE_DIR = PROJECT_ROOT / "data/uploads/images"
UPLOAD_LABEL_DIR = PROJECT_ROOT / "data/uploads/labels"
UPLOAD_LABEL_FILE = UPLOAD_LABEL_DIR / "labels.csv"

RETRAINING_REPORT = (
    PROJECT_ROOT / "outputs/retraining/candidate_evaluation.json"
)

DAMAGE_CODES = ["D00", "D10", "D20", "D40"]

CLASSES = {
    "D00": "Longitudinal cracks",
    "D10": "Transverse cracks",
    "D20": "Alligator cracks",
    "D40": "Potholes"
}


# ==================================================
# HEADER
# ==================================================

st.title("🛣️ RoadGuard AI")

st.subheader("Intelligent Road Infrastructure Monitoring System")

st.write(
    "Upload road images to identify possible road damage "
    "using a machine learning model."
)

st.divider()


# ==================================================
# API MONITORING
# ==================================================

st.header("System Monitoring")

col1, col2 = st.columns(2)

try:
    start_time = time.perf_counter()

    response = requests.get(
        f"{API_URL}/health",
        timeout=5
    )

    response.raise_for_status()

    elapsed = (time.perf_counter() - start_time) * 1000
    health = response.json()

    col1.metric("API Status", "Online")
    col2.metric(
        "Health Check Response",
        f"{elapsed:.2f} ms"
    )

    if not health.get("model_loaded", False):
        st.warning("API is online, but the model is not loaded.")

except requests.RequestException:
    col1.metric("API Status", "Offline")
    col2.metric("Health Check Response", "Unavailable")

    st.error(
        "Cannot connect to the RoadGuard API. "
        "Start the API server first."
    )

st.divider()


# ==================================================
# DATASET ANALYTICS
# ==================================================

st.header("Road Damage Dataset Analytics")

st.write(
    "Explore the distribution of road damage classes, "
    "damage combinations, and the number of damage classes "
    "present in each image."
)


@st.cache_data
def load_dataset_splits():

    train = pd.read_csv(
        PROJECT_ROOT / "data/processed/train.csv"
    )

    validation = pd.read_csv(
        PROJECT_ROOT / "data/processed/validation.csv"
    )

    test = pd.read_csv(
        PROJECT_ROOT / "data/processed/test.csv"
    )

    return pd.concat(
        [train, validation, test],
        ignore_index=True
    )


try:
    dataset_df = load_dataset_splits()

    class_names = {
        "D00": "Longitudinal cracks",
        "D10": "Transverse cracks",
        "D20": "Alligator cracks",
        "D40": "Potholes"
    }

    total_images = len(dataset_df)

    label_counts = dataset_df[DAMAGE_CODES].sum(axis=1)

    no_damage = int((label_counts == 0).sum())

    multiple_damage = int((label_counts > 1).sum())

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Images", f"{total_images:,}")

    col2.metric(
        "Multiple Damage Classes",
        f"{multiple_damage:,}"
    )

    col3.metric(
        "No Selected Target Class",
        f"{no_damage:,}"
    )

    # Chart 1: Class distribution

    st.subheader("1. Road Damage Class Distribution")

    class_counts = dataset_df[DAMAGE_CODES].sum()

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.bar(
        [class_names[c] for c in DAMAGE_CODES],
        class_counts.values
    )

    ax.set_ylabel("Number of Images")

    ax.set_title("Distribution of Road Damage Classes")

    ax.tick_params(axis="x", rotation=15)

    st.pyplot(fig)

    plt.close(fig)

    st.write(
        "Interpretation: Alligator cracks (D20) are the most "
        "frequent selected class, while transverse cracks (D10) "
        "are significantly underrepresented."
    )

    # Chart 2: Class co-occurrence

    st.subheader("2. Damage Class Co-occurrence")

    co_occurrence = (
        dataset_df[DAMAGE_CODES].astype(int).T
        @ dataset_df[DAMAGE_CODES].astype(int)
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    image = ax.imshow(co_occurrence.values)

    ax.set_xticks(range(len(DAMAGE_CODES)))

    ax.set_yticks(range(len(DAMAGE_CODES)))

    ax.set_xticklabels(DAMAGE_CODES)

    ax.set_yticklabels(DAMAGE_CODES)

    ax.set_title("Damage Class Co-occurrence")

    for i in range(len(DAMAGE_CODES)):
        for j in range(len(DAMAGE_CODES)):
            ax.text(
                j,
                i,
                str(co_occurrence.iloc[i, j]),
                ha="center",
                va="center"
            )

    fig.colorbar(
        image,
        ax=ax,
        label="Number of Images"
    )

    st.pyplot(fig)

    plt.close(fig)

    st.write(
        "Interpretation: D20 and D40 occur together most often. "
        "This shows that a single road image can contain "
        "multiple damage types."
    )

    # Chart 3: Number of damage classes per image

    st.subheader("3. Damage Classes Per Image")

    distribution = (
        label_counts.value_counts()
        .reindex(range(5), fill_value=0)
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(
        [
            "0 classes",
            "1 class",
            "2 classes",
            "3 classes",
            "4 classes"
        ],
        distribution.values
    )

    ax.set_ylabel("Number of Images")

    ax.set_title(
        "Number of Selected Damage Classes Per Image"
    )

    st.pyplot(fig)

    plt.close(fig)

    st.write(
        f"Interpretation: {multiple_damage:,} images contain "
        "more than one selected damage class. "
        "The system therefore uses multi-label classification."
    )

    st.caption(
        "Note: Zero selected classes means no annotation for "
        "D00, D10, D20, or D40. It does not necessarily mean "
        "the road is completely undamaged."
    )

except Exception as error:
    st.error(
        f"Could not load dataset analytics: {error}"
    )

st.divider()


# ==================================================
# IMAGE PREDICTION
# ==================================================

st.header("Road Damage Prediction")

uploaded_file = st.file_uploader(
    "Upload a road image",
    type=["jpg", "jpeg", "png"],
    key="prediction_image"
)

if uploaded_file is not None:

    st.image(
        uploaded_file,
        caption="Uploaded road image",
        use_container_width=True
    )

    if st.button(
        "Analyze Road Image",
        type="primary"
    ):

        with st.spinner("Analyzing image..."):

            try:
                uploaded_file.seek(0)

                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type
                    )
                }

                start_time = time.perf_counter()

                response = requests.post(
                    f"{API_URL}/predict",
                    files=files,
                    timeout=120
                )

                elapsed = (
                    time.perf_counter() - start_time
                ) * 1000

                if response.status_code == 200:

                    result = response.json()

                    predictions = result["predictions"]

                    st.success(
                        "Prediction completed successfully!"
                    )

                    st.metric(
                        "Prediction Response Time",
                        f"{elapsed:.2f} ms"
                    )

                    st.subheader("Prediction Results")

                    rows = []

                    for code, details in predictions.items():

                        rows.append({
                            "Damage Class": code,
                            "Description": CLASSES[code],
                            "Probability (%)": round(
                                details["probability"] * 100,
                                2
                            ),
                            "Threshold (%)": round(
                                details["threshold"] * 100,
                                2
                            ),
                            "Detected": (
                                "Yes"
                                if details["detected"]
                                else "No"
                            )
                        })

                    results_df = pd.DataFrame(rows)

                    st.dataframe(
                        results_df,
                        use_container_width=True,
                        hide_index=True
                    )

                    st.subheader("Prediction Probabilities")

                    chart_df = pd.DataFrame({
                        "Damage Class": [
                            CLASSES[code]
                            for code in predictions
                        ],
                        "Probability": [
                            details["probability"] * 100
                            for details in predictions.values()
                        ]
                    })

                    st.bar_chart(
                        chart_df.set_index("Damage Class")
                    )

                    detected = [
                        CLASSES[code]
                        for code, details in predictions.items()
                        if details["detected"]
                    ]

                    if detected:

                        st.warning(
                            "Possible road damage detected: "
                            + ", ".join(detected)
                        )

                    else:

                        st.info(
                            "No selected damage class crossed its "
                            "detection threshold."
                        )

                else:

                    st.error(
                        f"API error {response.status_code}: "
                        f"{response.text}"
                    )

            except requests.RequestException as error:

                st.error(
                    f"Could not contact the API: {error}"
                )

            except Exception as error:

                st.error(
                    f"Prediction failed: {error}"
                )


st.divider()


# ==================================================
# MODEL INFORMATION
# ==================================================

st.header("Model Information")

info1, info2, info3 = st.columns(3)

info1.metric("Architecture", "MobileNetV2")

info2.metric("Image Size", "160 × 160")

info3.metric("Damage Classes", "4")

st.caption(
    "RoadGuard AI is an experimental image-classification system. "
    "Its predictions should be reviewed by qualified road inspectors "
    "before infrastructure decisions are made."
)


# ==================================================
# BULK UPLOAD AND RETRAINING
# ==================================================

st.divider()

st.header("Bulk Upload and Model Retraining")

st.write(
    "Use existing verified road images and labels, "
    "or stage a new labelled dataset for retraining."
)


# --------------------------------------------------
# VALIDATE EXISTING UPLOADS ON FIRST DASHBOARD LOAD
# --------------------------------------------------

from src.retraining import validate_uploads

if "uploads_validated" not in st.session_state:

    st.session_state["uploads_validated"] = False

    st.session_state["validated_uploads_df"] = None

    try:

        existing_df = validate_uploads()

        if not existing_df.empty:

            st.session_state["uploads_validated"] = True

            st.session_state["validated_uploads_df"] = existing_df

    except Exception as error:

        st.session_state["uploads_validated"] = False

        st.session_state["validated_uploads_df"] = None

        st.session_state["upload_validation_error"] = str(error)


# --------------------------------------------------
# EXISTING VALIDATED DATASET
# --------------------------------------------------

existing_images = (
    list(UPLOAD_IMAGE_DIR.glob("*.jpg"))
    + list(UPLOAD_IMAGE_DIR.glob("*.jpeg"))
) if UPLOAD_IMAGE_DIR.exists() else []

labels_exist = UPLOAD_LABEL_FILE.is_file()

st.subheader("Existing Uploaded Dataset")

if st.session_state["uploads_validated"]:

    validated_df = st.session_state["validated_uploads_df"]

    st.success(
        "Existing uploaded images and labels have been validated."
    )

    st.metric(
        "Validated Uploaded Images",
        len(validated_df)
    )

    display_df = validated_df.copy()

    display_df.insert(
        0,
        "filename",
        display_df["image_path"].apply(
            lambda path: Path(path).name
        )
    )

    st.caption(
        "The existing files are being reused. "
        "You do not need to upload them again."
    )

    with st.expander("View validated upload labels"):

        st.dataframe(
            display_df[
                ["filename"] + DAMAGE_CODES
            ],
            use_container_width=True,
            hide_index=True
        )

elif existing_images or labels_exist:

    st.error(
        "Existing upload files were found, but validation did not "
        "complete successfully."
    )

    validation_error = st.session_state.get(
        "upload_validation_error"
    )

    if validation_error:
        st.error(validation_error)

    if st.button("Revalidate Existing Uploads"):

        try:

            validated_df = validate_uploads()

            if validated_df.empty:

                st.session_state["uploads_validated"] = False

                st.session_state["validated_uploads_df"] = None

                st.warning(
                    "No labelled uploaded images were found."
                )

            else:

                st.session_state["uploads_validated"] = True

                st.session_state["validated_uploads_df"] = validated_df

                st.session_state.pop(
                    "upload_validation_error",
                    None
                )

                st.success(
                    f"Successfully validated {len(validated_df)} images."
                )

        except Exception as error:

            st.error(
                f"Upload validation failed: {error}"
            )


# --------------------------------------------------
# NEW UPLOAD INTERFACE
# Only shown when there are no existing staged files.
# --------------------------------------------------

elif not existing_images and not labels_exist:

    st.subheader("Upload a New Dataset")

    uploaded_images = st.file_uploader(
        "Upload multiple road images (JPEG)",
        type=["jpg", "jpeg"],
        accept_multiple_files=True,
        key="bulk_images"
    )

    uploaded_labels = st.file_uploader(
        "Upload the labels CSV",
        type=["csv"],
        key="bulk_labels"
    )

    st.caption(
        "Required CSV columns: filename, D00, D10, D20, D40. "
        "Use 1 for present and 0 for absent."
    )

    if st.button("Validate and Stage Uploads"):

        if not uploaded_images or uploaded_labels is None:

            st.error(
                "Upload images and a labels CSV first."
            )

        else:

            staged_images = []

            labels_written = False

            label_path = UPLOAD_LABEL_FILE

            try:

                UPLOAD_IMAGE_DIR.mkdir(
                    parents=True,
                    exist_ok=True
                )

                UPLOAD_LABEL_DIR.mkdir(
                    parents=True,
                    exist_ok=True
                )

                filenames = [
                    file.name for file in uploaded_images
                ]

                if len(filenames) != len(set(filenames)):

                    raise ValueError(
                        "Duplicate image filenames uploaded."
                    )

                for filename in filenames:

                    if Path(filename).name != filename:

                        raise ValueError(
                            f"Invalid filename: {filename}"
                        )

                    if Path(filename).suffix.lower() not in [
                        ".jpg",
                        ".jpeg"
                    ]:

                        raise ValueError(
                            f"{filename}: only JPG/JPEG images "
                            "are accepted."
                        )

                    if (UPLOAD_IMAGE_DIR / filename).exists():

                        raise ValueError(
                            f"{filename} already exists. "
                            "Existing files will not be overwritten."
                        )

                if label_path.exists():

                    raise ValueError(
                        "A labels.csv already exists. "
                        "Existing staged data will not be overwritten."
                    )

                # Stage new images.

                for uploaded_image in uploaded_images:

                    image_path = (
                        UPLOAD_IMAGE_DIR / uploaded_image.name
                    )

                    image_path.write_bytes(
                        uploaded_image.getvalue()
                    )

                    staged_images.append(image_path)

                # Stage labels.

                label_path.write_bytes(
                    uploaded_labels.getvalue()
                )

                labels_written = True

                # Validate all staged files.

                validated_df = validate_uploads()

                if len(validated_df) != len(filenames):

                    raise ValueError(
                        "The validated image count does not "
                        "match the uploaded image count."
                    )

                validated_filenames = {
                    Path(path).name
                    for path in validated_df["image_path"]
                }

                if validated_filenames != set(filenames):

                    raise ValueError(
                        "The CSV filenames do not match "
                        "the uploaded images."
                    )

                st.session_state["uploads_validated"] = True

                st.session_state["validated_uploads_df"] = validated_df

                st.success(
                    "Upload validation successful!"
                )

                st.write(
                    "Validated images:",
                    len(validated_df)
                )

                st.rerun()

            except Exception as error:

                # Roll back only files created by this attempt.

                for image_path in staged_images:

                    image_path.unlink(missing_ok=True)

                if labels_written:

                    label_path.unlink(missing_ok=True)

                st.session_state["uploads_validated"] = False

                st.session_state["validated_uploads_df"] = None

                st.error(
                    f"Upload validation failed: {error}"
                )


# ==================================================
# RETRAINING CONTROL
# ==================================================

st.divider()

st.subheader("Model Retraining")

if st.session_state["uploads_validated"]:

    st.success(
        "Your existing labelled images are ready for retraining."
    )

    st.warning(
        "Retraining may take several minutes. "
        "Keep this dashboard open while the process runs."
    )

    if st.button(
        "Start Model Retraining",
        type="primary"
    ):

        try:

            from src.retraining import retrain

            from src.evaluate_candidate import (
                main as evaluate_candidate
            )

            with st.spinner(
                "Preparing data and retraining RoadGuard AI..."
            ):

                retrain()

            st.success(
                "Candidate model created successfully!"
            )

            with st.spinner(
                "Evaluating candidate against the current model..."
            ):

                evaluate_candidate()

            if not RETRAINING_REPORT.is_file():

                raise FileNotFoundError(
                    "Candidate evaluation report was not created."
                )

            with open(RETRAINING_REPORT) as file:

                report = json.load(file)

            st.subheader("Candidate Evaluation Results")

            current_f1 = report["current_metrics"]["macro_f1"]

            candidate_f1 = report["candidate_metrics"]["macro_f1"]

            col1, col2 = st.columns(2)

            col1.metric(
                "Current Model Macro F1",
                f"{current_f1:.4f}"
            )

            col2.metric(
                "Candidate Model Macro F1",
                f"{candidate_f1:.4f}",
                delta=f"{candidate_f1 - current_f1:+.4f}"
            )

            st.subheader("Per-Class Performance")

            comparison_rows = []

            for class_name, result in report[
                "class_comparisons"
            ].items():

                comparison_rows.append({

                    "Damage Class": class_name,

                    "Current F1": round(
                        result["current_f1"],
                        4
                    ),

                    "Candidate F1": round(
                        result["candidate_f1"],
                        4
                    ),

                    "Difference": round(
                        result["difference"],
                        4
                    ),

                    "Passed": result["passed"]

                })

            st.dataframe(
                comparison_rows,
                use_container_width=True
            )

            st.write(
                "Promotion status: Not performed"
            )

            if report["accepted_for_promotion"]:

                st.success(
                    "Candidate passed the evaluation criteria. "
                    "It is eligible for a separate promotion review."
                )

            else:

                st.warning(
                    "Candidate did not pass the evaluation criteria. "
                    "Keep the current production model."
                )

            st.info(
                "The current production model remains unchanged. "
                "Evaluation does not automatically promote candidates."
            )

        except Exception as error:

            st.error(
                f"Retraining or evaluation failed: {error}"
            )

else:

    st.info(
        "A valid labelled upload dataset is required "
        "before retraining can begin."
    )
