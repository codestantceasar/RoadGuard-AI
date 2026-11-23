from pathlib import Path

from locust import HttpUser, task, between


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SAMPLE_IMAGE = (
    PROJECT_ROOT
    / "locust/test_data/road_sample.jpg"
)

if not SAMPLE_IMAGE.is_file():
    raise FileNotFoundError(
        f"Load-test image not found: {SAMPLE_IMAGE}"
    )

IMAGE_BYTES = SAMPLE_IMAGE.read_bytes()


class RoadGuardUser(HttpUser):
    wait_time = between(1, 3)

    @task(1)
    def health_check(self):
        with self.client.get(
            "/health",
            name="/health",
            catch_response=True
        ) as response:

            if response.status_code != 200:
                response.failure("Health endpoint failed")
                return

            try:
                result = response.json()

                if result.get("model_loaded") is not True:
                    response.failure("Model is not loaded")
                else:
                    response.success()

            except ValueError:
                response.failure("Invalid health response")

    @task(4)
    def predict_road_damage(self):
        files = {
            "file": (
                "road_sample.jpg",
                IMAGE_BYTES,
                "image/jpeg"
            )
        }

        with self.client.post(
            "/predict",
            files=files,
            name="/predict",
            catch_response=True
        ) as response:

            if response.status_code != 200:
                response.failure(
                    f"Prediction failed: HTTP {response.status_code}"
                )
                return

            try:
                result = response.json()
                predictions = result.get("predictions", {})

                expected_classes = {"D00", "D10", "D20", "D40"}

                if set(predictions.keys()) != expected_classes:
                    response.failure("Unexpected prediction classes")
                else:
                    response.success()

            except ValueError:
                response.failure("Invalid prediction response")
