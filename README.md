# AI Monk — Real-Time Face Recognition & Attendance Monitoring Engine

A production-grade, low-latency, real-time employee identification and security/attendance monitoring system built for **Hybrid Edge-AI Architecture** and **NVIDIA RTX / Jetson GPU Servers** with a modular **React 18 + Vite** frontend.

---

## 🌟 Key Technical Highlights

### 1. Hybrid Edge-AI Web Architecture
- **Client-Side WASM Face Detection**: Runs the ultra-compact **Linzaer Ultra-Light 1MB RFB-320 ONNX** model directly inside the client browser via **WebAssembly SIMD** and **WebGL**. Detections take only **8–10 ms** per frame.
- **Zero Video Bandwidth Streaming**: Client devices (tablets/laptops/phones) do **not** stream continuous heavy video to the server. The client evaluates faces locally and only dispatches high-quality trigger crops when a prominent face is detected.

### 2. Smooth Proximity-Aware Motion Tracker
- **Generalized Association Tracker**: Combines **IoU (65%)** and **Center-Distance Proximity (35%)** with velocity motion extrapolation $(v_x, v_y)$ and exponential coordinate smoothing ($0.75 \text{det} + 0.25 \text{track}$).
- **Rock-Solid Motion Persistence**: Glides smoothly at the screen's native refresh rate (**60–120 FPS** on ProMotion displays) without jitter or box tearing during fast head movement, tilts, or walking.

### 3. Strict 120px Recognition Gate & High-Res Snapshots
- **120px Distance Filter**: Only triggers recognition when a face reaches $\ge 120\text{ px}$, preventing far-field background pedestrians from generating noise.
- **High-Quality Full-Frame Captures**: Dispatches a $224 \times 224$ face crop for embedding extraction along with a high-resolution annotated full camera frame ($95\%$ JPEG quality) with OpenCV adaptive gamma correction and contrast optimization.

### 4. Deep Face Recognition Pipeline (NVIDIA CUDA)
- **SCRFD 5-Point Landmark Alignment**: Aligns faces using affine transformation into standard $112 \times 112$ canonical geometry.
- **AdaFace IR-101 (512-d Embeddings)**: Quality-adaptive feature extraction running on CUDA GPU in **4–6 ms**.
- **FAISS Vector Index**: Cosine similarity matching ($\ge 0.53$, margin $\ge 0.04$) across 131+ enrolled employee embeddings in **0.027 ms**.

### 5. 30-Second Cooldown & State Preservation Rule
- **Initial Check-In (`CHECK_IN`)**: Verified with a solid **GREEN** box and recorded in the database.
- **Cooldown Window (0–30s)**: Subsequent appearances within 30 seconds retain solid **GREEN** and produce no duplicate attendance entries.
- **Re-Entry (`RE_ENTRY`)**: Appearances after 30 seconds turn **ORANGE** (`RE-ENTRY: Name [Score%]`) and log a new re-entry event. Matches during the subsequent 30s cooldown **remain ORANGE** without reverting to green.
- **Instant Mode Switch Bypass**: Switching between **ENTRY MODE** and **EXIT MODE** immediately clears cooldowns, allowing instant checkout (`CHECK_OUT`, Purple) or entry.

### 6. Multi-Device Real-Time Sync & Anti-Caching
- Strict HTTP anti-caching headers (`Cache-Control: no-cache, no-store`, timestamp salting) ensure all connected supervisor dashboards, guard tablets, and mobile phones update attendance feeds simultaneously in real-time.

---

## 🚀 Quick Setup & Run Guide

### Option A: Run via Docker Compose (Recommended Production Setup)

The system runs in Docker with PostgreSQL and NVIDIA GPU acceleration over HTTPS (Port `9001`):

```bash
# 1. Clone the repository and enter the directory
git clone https://github.com/your-org/ABLBL_Attendance.git
cd ABLBL_Attendance

# 2. Build the React frontend production bundle
npm run build

# 3. Start all services (PostgreSQL + Recognition Engine) with GPU support
docker compose up -d

# 4. Access the Live Dashboard:
# Open https://localhost:9001 or https://<server-ip>:9001 in your browser
# (Accept the one-time self-signed SSL certificate prompt)
```

---

### Option B: Local Development (Without Docker)

#### 1. Backend Setup
```bash
# Install Python dependencies
pip3 install -r requirements.txt

# Run backend server with HTTPS
python3 -m uvicorn src.api.app:app --host 0.0.0.0 --port 9001 --ssl-keyfile config/ssl.key --ssl-certfile config/ssl.crt
```

#### 2. Frontend Development Server (Hot Reloading)
```bash
# Install frontend dependencies
npm run install:frontend

# Start Vite dev server on port 3000 (proxies API to port 9001)
npm run dev
```

---

## 🛠️ Management & CLI Commands

| Task | Command |
| :--- | :--- |
| **Rebuild Frontend Bundle** | `npm run build` |
| **Restart Recognition Container** | `docker compose restart recognition_engine` |
| **View Live Container Logs** | `docker compose logs -f recognition_engine` |
| **Stop All Services** | `docker compose down` |
| **Enroll Employee Gallery (CLI)** | `PYTHONPATH=. python3 scripts/enroll_employees.py` |
| **Run Latency Benchmark** | `PYTHONPATH=. python3 scripts/benchmark.py` |
| **Generate PDF Architecture Report** | `python3 scripts/generate_architecture_report_pdf.py` |

---

## 📁 Project Architecture & Directory Layout

```
ABLBL_Attendance/
├── docker-compose.yml              # PostgreSQL + Recognition Engine container config
├── Dockerfile                      # CUDA 12.2 / Ubuntu 22.04 runtime container
├── package.json                    # Root npm scripts (build, dev, install)
├── requirements.txt                # Python backend dependencies
│
├── config/
│   ├── config.yaml                 # System thresholds, FAISS, camera parameters
│   ├── ssl.crt                     # SSL Certificate for HTTPS WebRTC
│   └── ssl.key                     # SSL Private Key
│
├── data/
│   ├── attendance_captures/        # High-res attendance capture snapshots
│   └── embeddings/                 # FAISS vector index files
│
├── Employees_Photo/                # Reference employee enrolled portrait photos
│
├── frontend/                       # Modular React 18 + Vite Application
│   ├── package.json                # React, Vite, ONNX Runtime Web, Lucide dependencies
│   ├── vite.config.js              # Vite configuration & API proxies
│   ├── index.html                  # HTML entry point with ONNX WASM loader
│   ├── dist/                       # Production bundled static SPA
│   └── src/
│       ├── main.jsx                # React DOM root
│       ├── App.jsx                 # App state orchestrator & polling loop
│       ├── App.css                 # Centralized dark-mode design system
│       ├── components/
│       │   ├── Header.jsx          # Top bar, Mode switch (ENTRY/EXIT)
│       │   ├── VideoPlayer.jsx     # Live WebRTC canvas viewport & loop
│       │   ├── ActivityFeed.jsx    # Right-side live feed with tabs (ALL/CHECK-IN/CHECK-OUT)
│       │   ├── ActivityCard.jsx    # Dual-photo card with sub-event pill toggles
│       │   ├── EmployeeModal.jsx   # Searchable 131+ employee gallery
│       │   ├── EnrollModal.jsx     # Multi-step employee enrollment modal
│       │   └── EventDetailModal.jsx# Full-resolution event comparison modal
│       ├── engine/
│       │   ├── UltraLightDetector.js # Linzaer 1MB RFB-320 ONNX detector & IoM NMS
│       │   ├── ClientIoUTracker.js  # Generalized IoU + Proximity motion tracker
│       │   ├── CropDispatcher.js    # 120px recognition gate & crop sender
│       │   └── CanvasRenderer.js    # HUD telemetry, bounding boxes & ID card popup
│       └── services/
│           └── api.js              # Anti-cached API client (/api/status, /api/mode, etc.)
│
├── models/
│   ├── kprpe_adaface.onnx          # AdaFace 512-d feature extraction model
│   ├── scrfd_2.5g_kps.onnx         # SCRFD 5-landmark alignment model
│   └── ultra_light/
│       └── version-RFB-320.onnx    # Linzaer Ultra-Light 1MB client ONNX model
│
├── scripts/
│   ├── benchmark.py                # Pipeline latency & throughput benchmark
│   ├── build_engines.py            # TensorRT engine compilation
│   ├── enroll_employees.py         # Batch employee photo embedding generator
│   └── generate_architecture_report_pdf.py # Generates formal PDF technical report
│
└── src/
    ├── api/
    │   ├── app.py                  # FastAPI application & SPA mounting
    │   └── routes.py               # REST API endpoints (/api/process_crop, /api/mode, etc.)
    ├── attendance/
    │   └── deduplication.py        # 30-second cooldown & deduplication controller
    ├── database/
    │   ├── models.py               # SQLAlchemy models (AttendanceEvent, Employee)
    │   ├── repository.py           # PostgreSQL / SQLite repository layer
    │   └── vector_store.py         # FAISS vector similarity index
    ├── detection/
    │   └── scrfd_detector.py       # SCRFD landmark alignment engine
    ├── recognition/
    │   └── adaface_recognizer.py   # AdaFace deep feature extractor
    └── pipeline.py                 # Core recognition pipeline & state machine
```

---

## 📡 REST API Reference

| Endpoint | Method | Payload / Params | Description |
| :--- | :---: | :--- | :--- |
| `/api/status` | `GET` | None | Returns system health, total enrolled count, and active mode (`ENTRY`/`EXIT`). |
| `/api/mode` | `GET` / `POST` | `{"mode": "ENTRY"}` | Gets or switches system mode between `ENTRY` and `EXIT` (clears cooldowns). |
| `/api/process_crop` | `POST` | JSON (`crop_base64`, `full_frame_base64`) | Receives client face crop, extracts AdaFace embedding, runs FAISS search, logs attendance event. |
| `/api/attendance/recent` | `GET` | `limit=50` | Returns recent attendance events with anti-cache headers. |
| `/api/employees` | `GET` | None | Returns list of all enrolled employees with photo URLs. |
| `/api/enroll` | `POST` | `multipart/form-data` | Enrolls a new employee (Name, ID, Department, Photo) and updates the FAISS index live. |

---

## 🛡️ License & Credits
Built for enterprise edge security and attendance monitoring. Proprietary and confidential.
