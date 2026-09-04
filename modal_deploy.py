"""
================================================================================
  AI MONK ATTENDANCE SYSTEM — MODAL SERVERLESS DEPLOYMENT (T4 GPU)
================================================================================
This script deploys the production FastAPI Face Recognition & Attendance Pipeline
to Modal using an NVIDIA T4 GPU and persistent cloud storage volumes.

Deploy to Modal:
  modal deploy modal_deploy.py

Test locally with live reload:
  modal serve modal_deploy.py
================================================================================
"""

import os
import shutil
import modal

# -----------------------------------------------------------------------------
# 1. MODAL APP & PERSISTENT VOLUMES
# -----------------------------------------------------------------------------
app = modal.App("ablbl-attendance")

# Persistent storage for attendance database, captures, embeddings, and reports
data_volume = modal.Volume.from_name("ablbl-attendance-data", create_if_missing=True)

# Persistent storage for reference employee photos
photos_volume = modal.Volume.from_name("ablbl-attendance-photos", create_if_missing=True)

# -----------------------------------------------------------------------------
# 2. CONTAINER IMAGE DEFINITION
# -----------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "libgl1",
        "libglib2.0-0",
        "libgomp1",
        "libsm6",
        "libxext6",
        "libxrender-dev",
    )
    .pip_install(
        "numpy==1.26.4",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "onnx>=1.14.0",
        "onnxruntime-gpu>=1.15.0",
        "nvidia-cudnn-cu12",
        "opencv-python>=4.8.0",
        "faiss-cpu>=1.7.4",
        "fastapi>=0.100.0",
        "uvicorn>=0.22.0",
        "sqlalchemy>=2.0.0",
        "psycopg2-binary>=2.9.0",
        "pyyaml>=6.0",
        "pillow>=9.5.0",
        "scipy>=1.11.0",
        "lap>=0.4.0",
        "filterpy>=1.4.5",
        "psutil>=5.9.0",
        "reportlab>=4.0.0",
        "python-multipart>=0.0.6",
    )
    # Bake application source code & configurations into container
    .add_local_dir("src", "/app/src")
    .add_local_dir("config", "/app/config")
    .add_local_dir("models", "/app/models")
    .add_local_dir("frontend/dist", "/app/frontend/dist")
    # Seed data baked in for instant initial volume hydration
    .add_local_dir("Employees_Photo", "/app/seed_Employees_Photo")
    .add_local_dir("data", "/app/seed_data")
)

# -----------------------------------------------------------------------------
# 3. SERVERLESS FASTAPI WEB SERVICE (T4 GPU ACCELERATED)
# -----------------------------------------------------------------------------
@app.cls(
    gpu="T4",
    image=image,
    volumes={
        "/app/data": data_volume,
        "/app/Employees_Photo": photos_volume,
    },
    scaledown_window=300,  # Keep warm for 5 minutes of inactivity
    timeout=600,
)
class AttendanceServer:
    @modal.enter()
    def startup(self):
        """
        One-time container initialization:
        1. Seeds persistent volumes if empty.
        2. Configures environment and working directory.
        3. Loads SCRFD, AdaFace, and FAISS gallery into GPU memory.
        """
        os.chdir("/app")
        os.environ["PYTHONPATH"] = "/app"

        # Hydrate persistent data volume if initial FAISS index is not yet present
        if not os.path.exists("/app/data/embeddings/faiss_index.bin") and os.path.exists("/app/seed_data"):
            print("[Modal Init] Seeding persistent data volume (embeddings, db, folders)...")
            for item in os.listdir("/app/seed_data"):
                src_item = os.path.join("/app/seed_data", item)
                dst_item = os.path.join("/app/data", item)
                if os.path.isdir(src_item) and not os.path.exists(dst_item):
                    shutil.copytree(src_item, dst_item)
                elif os.path.isfile(src_item) and not os.path.exists(dst_item):
                    shutil.copy2(src_item, dst_item)
            try:
                data_volume.commit()
            except Exception as e:
                print(f"[Modal Init] Data volume commit notice: {e}")

        # Hydrate persistent photos volume if empty
        if not os.path.exists("/app/Employees_Photo") or len(os.listdir("/app/Employees_Photo")) == 0:
            if os.path.exists("/app/seed_Employees_Photo"):
                print("[Modal Init] Seeding persistent photos volume...")
                os.makedirs("/app/Employees_Photo", exist_ok=True)
                for item in os.listdir("/app/seed_Employees_Photo"):
                    src_item = os.path.join("/app/seed_Employees_Photo", item)
                    dst_item = os.path.join("/app/Employees_Photo", item)
                    if os.path.isfile(src_item) and not os.path.exists(dst_item):
                        shutil.copy2(src_item, dst_item)
                try:
                    photos_volume.commit()
                except Exception as e:
                    print(f"[Modal Init] Photos volume commit notice: {e}")

        # Ensure all runtime directory structures exist
        os.makedirs("/app/Employees_Photo", exist_ok=True)
        os.makedirs("/app/data/attendance_captures", exist_ok=True)
        os.makedirs("/app/data/embeddings", exist_ok=True)
        os.makedirs("/app/data/enrollment_reports", exist_ok=True)

        # Initialize the production recognition pipeline & FastAPI application
        from src.api.app import create_app
        print("[Modal Init] Initializing RecognitionPipeline with CUDA T4 GPU...")
        self.fastapi_app = create_app(config_path="/app/config/config.yaml")
        print("[Modal Init] Recognition Engine ready to serve requests!")

    @modal.asgi_app()
    def web(self):
        """Exposes the FastAPI application as an HTTPS web endpoint."""
        return self.fastapi_app

# -----------------------------------------------------------------------------
# 4. STANDALONE MODAL CLI UTILITIES
# -----------------------------------------------------------------------------
@app.function(
    gpu="T4",
    image=image,
    volumes={
        "/app/data": data_volume,
        "/app/Employees_Photo": photos_volume,
    },
    timeout=1200,
)
def enroll_all_employees():
    """
    Utility task: Batch builds/rebuilds the FAISS gallery from photos in Employees_Photo.
    Run with:
      modal run modal_deploy.py::enroll_all_employees
    """
    os.chdir("/app")
    import yaml
    from src.enrollment.enroll_service import EnrollmentService

    with open("/app/config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("[Modal Task] Starting batch employee enrollment on T4 GPU...")
    enrollment_service = EnrollmentService(config)
    results = enrollment_service.enroll_from_directory("/app/Employees_Photo")

    # Persist the newly built FAISS index to the persistent volume
    data_volume.commit()
    photos_volume.commit()
    print(f"[Modal Task] Enrollment complete: {len(results)} employees processed and persisted.")
    return len(results)
