# AI Monk — Real-Time Face Recognition & Attendance Monitoring Engine

[![GitHub](https://img.shields.io/badge/GitHub-anas--work%2FABLBL__Attendance-blue?logo=github)](https://github.com/anas-work/ABLBL_Attendance)

A production-grade, low-latency, real-time employee identification and attendance monitoring system built on a **Hybrid Edge-AI Architecture** with **NVIDIA CUDA GPU** inference, a **React 18 + Vite** web frontend, and a native **Android client** (android_clientv6).

---

## 🌟 Key Technical Highlights

### 1. Hybrid Edge-AI Architecture
- **Client-Side Face Detection**: Runs the ultra-compact **Linzaer Ultra-Light RFB-320 (~1.3 MB)** ONNX model directly inside the browser via **WebAssembly / WebGL** — detections in **8–10 ms** per frame with zero server load.
- **1/3 Frame Decimation**: Client only runs inference on every 3rd frame. The tracker extrapolates the other 2 frames at native refresh rate — cutting client CPU/GPU load by ~66%.
- **Zero Video Streaming**: No raw video is sent to the server. Only quality-validated 224×224 JPEG face crops are dispatched.

### 2. Stable Proximity-Aware Motion Tracker
- **Generalized Association**: Combines **IoU (65%)** + **Center-Distance Proximity (35%)** with velocity extrapolation $(v_x, v_y)$.
- **Deadzone Jitter Filtering**: Ignores center shifts < 3.5px — prevents box jitter on stationary faces.
- **Heavy Dimension Smoothing**: 80% previous size / 20% new detection blend — eliminates bounding box size oscillation.
- **Box Calibration**: Top expanded, bottom trimmed, width expanded both sides — frames the face cleanly without including the chest.

### 3. Smart Recognition Gate
- **120px Size Gate**: Only triggers recognition when a face reaches ≥ 120px.
- **1-Second Settle Delay**: Waits 1 full second after the gate is crossed — lets the person stop moving, dramatically reducing false negatives.
- **5-Frame Controlled Sampling**: Sends exactly 5 frames at ~600ms intervals within a 3-second window — never floods the API, maximises accuracy.
- **State Machine**: `WAITING_FOR_SIZE → SETTLING → RECOGNIZING → MATCHED / NOT_RECOGNIZED`

### 4. Server-Side Recognition Pipeline (NVIDIA CUDA)
- **SCRFD 5-Point Landmark Alignment**: Affine transformation to 112×112 canonical geometry.
- **KPRPe + AdaFace IR-101 (512-d)**: Quality-adaptive GPU embedding extraction in **4–6 ms**.
- **FAISS Flat Index**: Cosine similarity search (threshold ≥ 0.50, margin gate ≥ 0.04) across enrolled embeddings in **< 0.05 ms**.

### 5. Entry/Exit State Machine with Cooldowns
- `CHECK_IN` → **Green** box, recorded on first verified entry.
- `RE_ENTRY` → **Orange** box, logged after cooldown expires.
- `CHECK_OUT` → **Purple** box, triggered in EXIT MODE.
- Mode switch immediately clears cooldowns for instant re-recognition.

### 6. Android App (android_clientv6)
- Full UI parity with the web app — same stat cards, filter tabs, activity feed, enroll dialog.
- **LiteRT 1.4.0** (Google's TFLite successor) — 16KB page-size aligned for Android 15.
- **CameraX 1.4.0** — fixes `libimage_processing_util_jni.so` alignment.
- Edge-to-edge / notch-aware layout, transparent status/nav bars.
- Identical 1/3 decimation, settle delay, and 5-frame sampling logic as the web app.

---

## ⚙️ Prerequisites

| Requirement | Version |
|---|---|
| NVIDIA GPU + CUDA | 12.1+ |
| Docker + Docker Compose | Latest |
| NVIDIA Container Toolkit | For GPU passthrough |
| Node.js | 18+ |
| Python | 3.10+ |
| Android Studio (for app) | Hedgehog+ |

---

## 🚀 Quick Setup & Run

### Step 0: Clone & Download the Large Model

```bash
git clone https://github.com/anas-work/ABLBL_Attendance.git
cd ABLBL_Attendance
```

> **⚠️ The AdaFace recognition model is NOT included in the repo** (167MB > GitHub's 100MB limit).
> Download it separately and place it at `models/kprpe_adaface.onnx`:
>
> ```bash
> # Option 1: Copy from your existing server
> scp user@your-server:/path/to/kprpe_adaface.onnx models/
>
> # Option 2: Download from your shared storage location
> # (Ask the team for the model file link)
> ```

### Step 1: Generate SSL Certificates

The system serves over HTTPS (required for browser camera access via `getUserMedia`).

```bash
mkdir -p config
openssl req -x509 -newkey rsa:4096 -keyout config/ssl.key \
  -out config/ssl.crt -days 365 -nodes \
  -subj "/CN=attendance-server"
```

### Step 2: Build the React Frontend

```bash
npm run build
# This runs: cd frontend && npm install && npm run build
# Output goes to: frontend/dist/
```

### Step 3: Run via Docker Compose (Recommended)

```bash
# Start PostgreSQL + Recognition Engine (GPU accelerated, HTTPS on port 9001)
docker compose up -d

# View live logs
docker compose logs -f recognition_engine

# Access the dashboard
# https://localhost:9001
# https://<your-server-ip>:9001
# (Accept the one-time self-signed SSL warning on first load)
```

---

## 🛠️ Local Development (Without Docker)

### Backend

```bash
# Install Python dependencies
pip3 install -r requirements.txt

# Set PYTHONPATH and run the FastAPI server with HTTPS
PYTHONPATH=. python3 -m uvicorn src.api.app:app \
  --host 0.0.0.0 --port 9001 \
  --ssl-keyfile config/ssl.key \
  --ssl-certfile config/ssl.crt
```

### Frontend (Hot Reloading)

```bash
cd frontend
npm install
npm run dev
# Vite dev server at http://localhost:3000
# API calls are proxied to https://localhost:9001
```

---

## 📱 Android App (android_clientv6)

The Android client has full feature parity with the web app.

### Requirements
- Android Studio Hedgehog or newer
- Android SDK 35 (Android 15 target)
- `minSdk = 26` (Android 8.0+)

### Setup
1. Open `android_clientv6/` in Android Studio
2. Update the server IP in `app/src/main/java/com/aimonk/attendance/network/ApiService.kt`:
   ```kotlin
   private const val BASE_URL = "https://YOUR_SERVER_IP:9001/"
   ```
3. Sync Gradle → the app uses **LiteRT 1.4.0** (16KB page-aligned, Android 15 compatible)
4. Build & run on device or emulator (API 26+)

### Key Features
- Same detection engine: UltraLight TFLite running on-device via LiteRT + GPU delegate
- Same pipeline: 1/3 frame decimation, 1s settle delay, 5-frame recognition sampling
- SSL-trusting OkHttpClient — works with self-signed server certs
- Edge-to-edge UI with notch / display cutout support

---

## 📁 Project Structure

```
ABLBL_Attendance/
├── docker-compose.yml              # PostgreSQL + Recognition Engine
├── Dockerfile                      # CUDA 12.1 / Ubuntu 22.04 runtime
├── requirements.txt                # Python backend dependencies
├── package.json                    # Root npm scripts (build, dev)
│
├── config/
│   ├── config.yaml                 # Thresholds, models, DB, camera config
│   ├── ssl.crt                     # SSL certificate (not in repo — generate locally)
│   └── ssl.key                     # SSL private key (not in repo — generate locally)
│
├── frontend/                       # React 18 + Vite SPA
│   ├── src/
│   │   ├── App.jsx                 # State orchestrator & polling loop
│   │   ├── components/             # Header, VideoPlayer, ActivityFeed, Modals
│   │   ├── engine/
│   │   │   ├── UltraLightDetector.js  # ONNX face detector (WASM/WebGL)
│   │   │   ├── ClientIoUTracker.js    # IoU + Proximity tracker w/ jitter filter
│   │   │   ├── CropDispatcher.js      # Gate, settle, 5-frame sampling logic
│   │   │   └── CanvasRenderer.js      # HUD, bounding boxes, ID card popup
│   │   └── services/api.js         # Anti-cached REST client
│   └── vite.config.js
│
├── android_clientv6/               # Native Android App (Kotlin + CameraX + LiteRT)
│   └── app/src/main/
│       ├── java/com/aimonk/attendance/
│       │   ├── MainActivity.kt     # Camera pipeline, UI state, insets handling
│       │   ├── engine/             # UltraLightDetector, IoUTracker, CropDispatcher
│       │   ├── ui/                 # Adapters, Dialogs (Enroll, Employees, Comparison)
│       │   └── network/ApiService.kt  # OkHttp3 + SSL-trust client
│       └── res/                    # Layouts, drawables, themes
│
├── models/
│   ├── kprpe_adaface.onnx          # ⚠️ NOT IN REPO (167MB) — download separately
│   ├── scrfd_2.5g_kps.onnx         # SCRFD 5-landmark detection model
│   └── ultra_light/
│       └── version-RFB-320.onnx    # Client-side 1.3MB face detector
│
├── src/                            # Python backend
│   ├── api/
│   │   ├── app.py                  # FastAPI app factory & static mounts
│   │   └── routes.py               # All REST endpoints
│   ├── pipeline.py                 # Core recognition pipeline & state machine
│   ├── detection/scrfd_detector.py # SCRFD ONNX landmark detector
│   ├── recognition/kprpe_adaface.py# AdaFace GPU embedding extractor
│   ├── search/faiss_index.py       # FAISS flat index (cosine similarity)
│   ├── attendance/deduplication.py # Cooldown & deduplication controller
│   ├── enrollment/enroll_service.py# Employee enrollment pipeline
│   ├── database/                   # SQLAlchemy models + PostgreSQL/SQLite repo
│   └── video/                      # File, Camera, RTSP video sources
│
├── scripts/
│   ├── enroll_employees.py         # Batch enroll from Employees_Photo/
│   ├── benchmark.py                # Pipeline latency benchmark
│   └── build_engines.py            # TensorRT engine compiler
│
└── tests/
    ├── test_components.py
    └── test_live_pipeline.py
```

---

## 📡 REST API Reference

| Endpoint | Method | Description |
|---|:---:|---|
| `/api/status` | `GET` | System health, enrolled count, present/absent/unknown, active mode |
| `/api/mode` | `GET`/`POST` | Get or switch ENTRY/EXIT mode (clears cooldowns) |
| `/api/process_crop` | `POST` | Receive face crop (base64 JSON), run AdaFace + FAISS, log attendance |
| `/api/record_unknown` | `POST` | Flag an unrecognized person after 5 failed attempts |
| `/api/attendance/recent` | `GET` | Last 500 attendance events |
| `/api/attendance/flush` | `POST` | Clear all events and reset presence state |
| `/api/employees` | `GET` | Full enrolled employee directory with presence indicators |
| `/api/enroll` | `POST` | Enroll new employee (name, ID, photo) — updates FAISS index live |
| `/api/employees/{id}` | `DELETE` | Remove employee from gallery, DB, and disk |
| `/video_feed` | `GET` | MJPEG stream of annotated server-side video (file mode) |

---

## 🛠️ Management Commands

```bash
# Rebuild frontend after code changes
npm run build

# Restart recognition engine (after config/model changes)
docker compose restart recognition_engine

# View live logs
docker compose logs -f recognition_engine

# Stop everything
docker compose down

# Batch enroll employees from Employees_Photo/
# (Photos must be named: "Firstname Lastname EMP_ID.jpg")
PYTHONPATH=. python3 scripts/enroll_employees.py

# Run pipeline latency benchmark
PYTHONPATH=. python3 scripts/benchmark.py

# Connect to PostgreSQL
docker exec -it attendance_postgres psql -U postgres -d attendance_db

# Query recent attendance
docker exec -it attendance_postgres psql -U postgres -d attendance_db \
  -c "SELECT employee_id, event_type, timestamp FROM attendance_events ORDER BY timestamp DESC LIMIT 20;"
```

---

## 🔧 Configuration (`config/config.yaml`)

Key settings to customize:

```yaml
hardware:
  device: "cuda"          # "cuda" or "cpu"
  precision: "fp16"       # fp16 for speed, fp32 for accuracy

recognition:
  match_threshold: 0.50   # Cosine similarity threshold (raise to reduce false positives)
  unknown_threshold: 0.24 # Below this = UNKNOWN

attendance:
  cooldown_seconds: 5     # Seconds before a repeat entry is logged again

database:
  url: "postgresql://postgres:postgres@localhost:5432/attendance_db"
  sqlite_fallback_url: "sqlite:///data/attendance.db"  # Used if Postgres unavailable
```

---

## 🛡️ License & Credits

Built for enterprise edge security and attendance monitoring by **AI Monk**.  
Proprietary and confidential — © 2026 ABLBL / AI Monk. All rights reserved.
