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
import sys
import shutil
import modal

# Ensure /app is in sys.path in local and remote python environments
if "/app" not in sys.path:
    sys.path.insert(0, "/app")

# -----------------------------------------------------------------------------
# 1. MODAL APP & PERSISTENT VOLUMES
# -----------------------------------------------------------------------------
app = modal.App("ablbl-attendance")

# Persistent storage for attendance database, captures, embeddings, and reports
data_volume = modal.Volume.from_name("ablbl-attendance-data", create_if_missing=True)

# -----------------------------------------------------------------------------
# 2. CONTAINER IMAGE DEFINITION
# -----------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.10")
    .env({
        "PYTHONPATH": "/app",
        "BUILD_VERSION": "v2.0-fast-local-io",
        "LD_LIBRARY_PATH": "/usr/local/lib/python3.10/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.10/site-packages/nvidia/cufft/lib:/usr/local/lib/python3.10/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.10/site-packages/nvidia/curand/lib:/usr/local/lib/python3.10/site-packages/nvidia/cusolver/lib:/usr/local/lib/python3.10/site-packages/nvidia/cusparse/lib:/usr/local/lib/python3.10/site-packages/nvidia/nvjitlink/lib:/usr/local/cuda/lib64"
    })
    .apt_install(
        "build-essential",
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
        "onnxruntime-gpu>=1.17.0",
        "nvidia-cudnn-cu12",
        "nvidia-cufft-cu12",
        "nvidia-cublas-cu12",
        "nvidia-cuda-runtime-cu12",
        "nvidia-curand-cu12",
        "nvidia-cusolver-cu12",
        "nvidia-cusparse-cu12",
        "opencv-python>=4.8.0",
        "faiss-cpu>=1.7.4",
        "fastapi>=0.100.0",
        "httpx>=0.24.0",
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
    .add_local_dir("Employees_Video", "/app/Employees_Video")
    .add_local_dir("Employees_Photo", "/app/Employees_Photo")
    .add_local_dir("data/embeddings", "/app/local_embeddings")
)

# -----------------------------------------------------------------------------
# 3. SERVERLESS FASTAPI WEB SERVICE (T4 GPU ACCELERATED)
# -----------------------------------------------------------------------------
@app.cls(
    gpu="T4",
    image=image,
    volumes={
        "/app/data": data_volume,
    },
    scaledown_window=300,  # Keep warm for 5 minutes of inactivity
    timeout=600,
    max_containers=1,
)
class AttendanceServer:
    @modal.enter()
    def startup(self):
        """
        One-time container initialization:
        1. Ensures local working directory & sys.path.
        2. Ensures persistent directories exist for captures & database.
        3. Loads SCRFD, AdaFace, and FAISS gallery into GPU memory.
        """
        import sys
        if "/app" not in sys.path:
            sys.path.insert(0, "/app")
        os.chdir("/app")
        os.environ["PYTHONPATH"] = "/app"

        # Ensure runtime directory structures exist
        os.makedirs("/app/data/attendance_captures", exist_ok=True)
        os.makedirs("/app/data/enrolled_photos", exist_ok=True)
        os.makedirs("/app/data/embeddings", exist_ok=True)
        os.makedirs("/app/data/enrollment_reports", exist_ok=True)

        # Seed FAISS embeddings from the baked container image ONLY if the volume
        # does not yet have a gallery. Once employees are enrolled via the web UI,
        # their data is preserved on the persistent volume across all restarts.
        if not os.path.exists("/app/data/embeddings/faiss_index.bin") and os.path.exists("/app/local_embeddings"):
            for item in os.listdir("/app/local_embeddings"):
                src = os.path.join("/app/local_embeddings", item)
                dst = os.path.join("/app/data/embeddings", item)
                shutil.copy2(src, dst)
            print("[Modal Init] Seeded FAISS embeddings from baked container image (first boot only).")
        else:
            print(f"[Modal Init] Using existing FAISS gallery from persistent volume.")

        # Initialize the production recognition pipeline & FastAPI application
        from src.api.app import create_app
        print("[Modal Init] Initializing RecognitionPipeline with CUDA T4 GPU...")
        self.fastapi_app = create_app(config_path="/app/config/config.yaml")
        print("[Modal Init] Recognition Engine ready to serve requests!")

    @modal.asgi_app(label="ablbl-attendance")
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
    },
    timeout=1200,
)
def enroll_all_employees():
    """
    Utility task: Batch builds/rebuilds the FAISS gallery from photos in Employees_Photo.
    Run with:
      modal run modal_deploy.py::enroll_all_employees
    """
    import sys
    if "/app" not in sys.path:
        sys.path.insert(0, "/app")
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
    print(f"[Modal Task] Enrollment complete: {len(results)} employees processed and persisted.")
    return len(results)


# -----------------------------------------------------------------------------
# 5. DEBUG & MAINTENANCE UTILITIES
# -----------------------------------------------------------------------------
@app.function(
    image=image,
    volumes={"/app/data": data_volume},
    timeout=120,
)
def inspect_volume():
    """Inspect what's currently stored on the persistent volume.
    Run: modal run modal_deploy.py::inspect_volume
    """
    import json
    print("\n=== FAISS Gallery (metadata.json) ===")
    meta_path = "/app/data/embeddings/metadata.json"
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            entries = json.load(f)
        print(f"Total entries: {len(entries)}")
        for e in entries:
            print(f"  {e.get('employee_id'):15} | {e.get('name'):30} | {e.get('filename')}")
    else:
        print("  (no metadata.json found)")

    print("\n=== data/enrolled_photos (persistent volume) ===")
    ep_dir = "/app/data/enrolled_photos"
    if os.path.exists(ep_dir):
        files = os.listdir(ep_dir)
        print(f"Total files: {len(files)}")
        for fname in sorted(files):
            fpath = os.path.join(ep_dir, fname)
            size = os.path.getsize(fpath)
            print(f"  {fname} ({size} bytes)")
    else:
        print("  (directory does not exist)")


@app.function(
    image=image,
    volumes={"/app/data": data_volume},
    timeout=120,
)
def wipe_enrolled_photos():
    """Deletes ALL files from data/enrolled_photos on the persistent volume.
    Use this to clear stale web-enrolled photos before re-enrolling cleanly.
    Run: modal run modal_deploy.py::wipe_enrolled_photos
    """
    ep_dir = "/app/data/enrolled_photos"
    deleted = 0
    if os.path.exists(ep_dir):
        for fname in os.listdir(ep_dir):
            fpath = os.path.join(ep_dir, fname)
            os.remove(fpath)
            deleted += 1
            print(f"  Deleted: {fname}")
    data_volume.commit()
    print(f"\nWiped {deleted} file(s) from {ep_dir}. Volume committed.")
    return deleted


@app.function(
    image=image,
    volumes={"/app/data": data_volume},
    timeout=120,
)
def reset_gallery_to_baseline():
    """Resets the FAISS gallery on the persistent volume back to the baked
    container image baseline. This removes any web-enrolled entries whose
    photos are now stale. The enrolled_photos directory is NOT touched.
    Run: modal run modal_deploy.py::reset_gallery_to_baseline
    """
    import json
    embeddings_dst = "/app/data/embeddings"
    os.makedirs(embeddings_dst, exist_ok=True)
    copied = 0
    for item in os.listdir("/app/local_embeddings"):
        src = os.path.join("/app/local_embeddings", item)
        dst = os.path.join(embeddings_dst, item)
        shutil.copy2(src, dst)
        copied += 1
        print(f"  Restored: {item}")
    data_volume.commit()
    print(f"\nReset {copied} file(s) in FAISS gallery to baseline. Volume committed.")
    print("On next container start, _restore_web_enrolled_employees will re-add photos from enrolled_photos.")
    return copied
