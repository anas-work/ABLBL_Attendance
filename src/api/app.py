import os
import yaml
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import src.api.routes as routes
from src.pipeline import RecognitionPipeline

def create_app(config_path: str = "config/config.yaml") -> FastAPI:
    app = FastAPI(title="Real-Time Face Recognition & Attendance Monitoring Engine")

    # Security header middleware to grant camera permissions for WebRTC live video
    @app.middleware("http")
    async def add_security_headers(request, call_next):
        response = await call_next(request)
        response.headers["Permissions-Policy"] = "camera=(self), microphone=()"
        return response

    # Ensure storage directories exist
    os.makedirs("Employees_Photo", exist_ok=True)
    os.makedirs("data/attendance_captures", exist_ok=True)
    os.makedirs("data/enrolled_photos", exist_ok=True)
    os.makedirs("data/embeddings", exist_ok=True)
    os.makedirs("data/enrollment_reports", exist_ok=True)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Initialize recognition pipeline
    pipeline = RecognitionPipeline(config=config)

    # Re-embed any web-enrolled employees from the persistent volume (data/enrolled_photos)
    # that are not yet in the FAISS gallery
    try:
        pipeline._restore_web_enrolled_employees()
    except Exception as restore_err:
        print(f"[App Init] Warning: could not restore web-enrolled employees: {restore_err}")
    
    # Clear DB events on startup for an empty activity feed
    if pipeline.db_repo:
        pipeline.db_repo.clear_all_events()

    routes.pipeline_instance = pipeline
    app.include_router(routes.router)

    # Mount static files for captures and client ONNX models
    app.mount("/captures", StaticFiles(directory="data/attendance_captures"), name="captures")
    app.mount("/models", StaticFiles(directory="models"), name="models")

    # Mount Vite bundled assets
    dist_assets_dir = "frontend/dist/assets"
    if os.path.exists(dist_assets_dir):
        app.mount("/assets", StaticFiles(directory=dist_assets_dir), name="assets")

    @app.get("/", response_class=HTMLResponse)
    async def read_index():
        dist_index = "frontend/dist/index.html"
        index_path = dist_index if os.path.exists(dist_index) else "frontend/index.html"
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()

    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    ssl_key = "config/ssl.key" if os.path.exists("config/ssl.key") else None
    ssl_cert = "config/ssl.crt" if os.path.exists("config/ssl.crt") else None
    uvicorn.run(app, host="0.0.0.0", port=9001, ssl_keyfile=ssl_key, ssl_certfile=ssl_cert)
