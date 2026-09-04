import os
import yaml
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import src.api.routes as routes
from src.pipeline import RecognitionPipeline
from src.video.file_source import FileVideoSource
from src.video.camera_source import CameraVideoSource

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

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # ------------------------------------------------------------------ #
    #  Video source selection: live camera  vs.  pre-recorded file        #
    #  Priority: env var LIVE_MODE=1  >  config video.source_type=camera  #
    # ------------------------------------------------------------------ #
    video_cfg = config.get("video", {})
    live_mode = (
        os.environ.get("LIVE_MODE", "").strip().lower() in ("1", "true", "yes")
        or video_cfg.get("source_type", "file") == "camera"
    )

    if live_mode:
        device_id_raw = os.environ.get("CAMERA_DEVICE_ID", str(video_cfg.get("camera_device_id", 0)))
        # Accept integer index or a GStreamer pipeline string
        try:
            device_id = int(device_id_raw)
        except ValueError:
            device_id = device_id_raw  # GStreamer / V4L2 string path

        try:
            video_source = CameraVideoSource(
                device_id=device_id,
                width=video_cfg.get("frame_width", 1280),
                height=video_cfg.get("frame_height", 720),
                target_fps=float(video_cfg.get("target_fps", 30.0)),
            )
            print(f"[VideoSource] LIVE CAMERA mode — device={device_id} "
                  f"{video_source.resolution[0]}x{video_source.resolution[1]} @ {video_source.fps:.1f} FPS")
        except Exception as cam_err:
            print(f"[VideoSource] WARNING: Could not open camera device {device_id!r}: {cam_err}")
            print("[VideoSource] Falling back to FILE mode.")
            video_source = FileVideoSource(video_cfg.get("source", "Employees_Video/Live_Feed.mp4"), loop=True)
    else:
        video_source = FileVideoSource(video_cfg.get("source", "Employees_Video/Live_Feed.mp4"), loop=True)
        print(f"[VideoSource] FILE mode — {video_cfg.get('source', 'Employees_Video/Live_Feed.mp4')}")

    # Initialize recognition pipeline
    pipeline = RecognitionPipeline(config=config, video_source=video_source)

    # Re-embed any web-enrolled employees from the persistent volume (data/enrolled_photos)
    # that are not yet in the FAISS gallery (because startup always seeds from baked image).
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
