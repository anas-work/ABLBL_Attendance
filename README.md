# AI Monk — Real-Time Face Recognition & Attendance Monitoring Engine (Server Edition)

[![GitHub](https://img.shields.io/badge/GitHub-anas--work%2FABLBL__Attendance-blue?logo=github)](https://github.com/anas-work/ABLBL_Attendance)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)](https://www.docker.com/)
[![NVIDIA CUDA](https://img.shields.io/badge/NVIDIA-CUDA_12.2-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react)](https://react.dev/)

A production-grade, low-latency, real-time employee identification and attendance monitoring system built on a **Server-Side AI Architecture** with **NVIDIA CUDA GPU** acceleration, a modern **React 18 + Vite** web dashboard, and a **FastAPI + PostgreSQL** persistence backend.

---

## 🌟 Key Technical Architecture

### 1. Zero-Server-Load Client Detection
- **Browser-Side Face Detection**: Runs the ultra-compact **Linzaer Ultra-Light RFB-320 (~1.3 MB)** ONNX model directly inside the client browser using **WebAssembly / WebGL** — detections in **8–10 ms** per frame.
- **1/3 Frame Decimation**: Client runs inference on every 3rd frame. The tracker extrapolates remaining frames at the native display refresh rate — cutting client CPU/GPU overhead by ~66%.
- **Bandwidth Optimization**: Raw continuous video streams are **never** streamed to the server. Only quality-validated 224×224 JPEG face crops are dispatched when a person approaches the camera.

### 2. Proximity-Aware Motion Tracker
- **Generalized Spatial Association**: Blends **IoU (65%)** and **Center-Distance Proximity (35%)** with velocity extrapolation $(v_x, v_y)$.
- **Deadzone Jitter Filter**: Suppresses center shifts < 3.5 px to eliminate bounding box jitter on stationary faces.
- **Heavy Dimension Smoothing**: Blends 80% previous box size / 20% new detection to prevent rapid size oscillations.
- **Box Calibration**: Expanded top, trimmed bottom, and adjusted width to frame faces cleanly.

### 3. Smart Recognition Gate
- **120px Size Gate**: Triggers recognition only when a face bounding box reaches $\ge 120\text{ px}$.
- **1-Second Settle Delay**: Pauses 1 full second after crossing the size gate to allow the person to stabilize, drastically minimizing motion blur and false negatives.
- **5-Frame Controlled Sampling**: Dispatches at most 5 frames spaced at ~600 ms intervals over a 3-second evaluation window — eliminating server flooding while maximizing recognition accuracy.
- **State Progression**: `APPROACH CAMERA (<120px) → HOLD STEADY (Settle) → ANALYZING (1-5/5) → MATCHED / NOT_RECOGNIZED`

### 4. Server-Side Recognition Pipeline (NVIDIA CUDA)
- **SCRFD 5-Point Landmark Alignment**: 5 facial keypoints detected and mapped to 112×112 canonical geometry via affine similarity transformation.
- **KPRPe + AdaFace IR-101 (512-d)**: Quality-adaptive GPU embedding extraction in **3–5 ms**.
- **FAISS Flat Index**: Cosine similarity search (match threshold $\ge 0.53$, margin gate $\ge 0.04$) across registered vectors in **< 0.1 ms**.
- **Dual Persistence Strategy**: Primary PostgreSQL database with automatic, zero-downtime SQLite fallback (`data/attendance.db`).

---

## 🔄 Employee Data Lifecycle: Enrollment & Complete Deletion

The system implements a deterministic, 360-degree data lifecycle for employee registration and removal across vector indexes, physical storage, relational databases, and in-memory state machines:

```
                          ┌──────────────────────────┐
                          │   POST /api/enroll       │
                          └─────────────┬────────────┘
                                        │
             ┌──────────────────────────┼──────────────────────────┐
             ▼                          ▼                          ▼
   [ FAISS Vector Index ]     [ Physical Disk Photos ]    [ Database & Memory ]
   • 5-point landmark align   • Employees_Photo/          • PostgreSQL & SQLite
   • 512-d AdaFace embedding  • data/enrolled_photos/     • in-memory photo cache
   • metadata.json + bin      • {Name} {ID}.jpg           • live presence enable
```

```
                          ┌──────────────────────────┐
                          │ DELETE /api/employees/ID │
                          └─────────────┬────────────┘
                                        │
             ┌──────────────────────────┼────────────────────────────┐
             ▼                          ▼                            ▼
   [ FAISS Vector Gallery ]   [ Physical Disk Storage ]    [ Databases & In-Memory ]
   • Removes vector embedding • Deletes Employees_Photo/   • Deletes from employees
   • Rebuilds IndexFlatIP     • Deletes data/enrolled/     • Deletes from enrollments
   • Saves faiss_index.bin    • Deletes attendance captures• Deletes attendance_events
   • Saves metadata.json        from data/attendance_      • Deletes recognition_events
                                captures/                  • Clears presence & cooldowns
```

### Complete Deletion Scope (`DELETE /api/employees/{id}`)
1. **FAISS Vector Gallery**: Vector embeddings and metadata entries are purged; `faiss_index.bin` and `metadata.json` are reconstructed and saved to disk.
2. **In-Memory Caches**: Clears `employee_photos`, `employee_photo_paths`, `globally_marked_present_employees`, `last_event_type`, `deduplicator.last_recorded`, and `track_identities`.
3. **Physical Disk Files**:
   - `Employees_Photo/` (exact filename and matching ID pattern).
   - `data/enrolled_photos/` (exact filename and matching ID pattern).
   - `data/attendance_captures/` (all snapshot capture images for that employee).
4. **Relational Database**:
   - Deletes matching records from `attendance_events`.
   - Deletes matching records from `recognition_events`.
   - Deletes matching records from `enrollments`.
   - Deletes matching record from `employees`.

---

## ⚙️ Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| NVIDIA GPU + CUDA | 12.0+ | Compute capability 7.0+ (RTX / T4 / A4000 / L4 / A100) |
| Docker & Docker Compose | Latest | Standard container runtime |
| NVIDIA Container Toolkit | Latest | GPU device passthrough into Docker |
| Node.js | 18+ | For building frontend assets |
| Python | 3.10+ | For development outside Docker |

---

## 🚀 Quick Setup & Run (Docker Compose)

### Step 0: Clone & Place Model Files

```bash
git clone https://github.com/anas-work/ABLBL_Attendance.git
cd ABLBL_AttendanceV1
```

> **⚠️ The AdaFace model (`kprpe_adaface.onnx`, 167MB) is not bundled in Git.**
> Place it at `models/kprpe_adaface.onnx`:
> ```bash
> # Copy from backup/server
> cp /path/to/kprpe_adaface.onnx models/
> ```

### Step 1: Generate SSL Certificates

HTTPS is required by modern browsers to grant `getUserMedia` webcam access across network interfaces:

```bash
mkdir -p config
openssl req -x509 -newkey rsa:4096 -keyout config/ssl.key \
  -out config/ssl.crt -days 365 -nodes \
  -subj "/CN=attendance-server"
```

### Step 2: Build the Frontend Production Bundle

```bash
npm run build
# Generates compiled assets in frontend/dist/
```

### Step 3: Start Services with Docker Compose

```bash
# Start PostgreSQL database + GPU-accelerated Recognition Engine (Port 9001)
docker compose up -d

# View real-time container logs
docker compose logs -f recognition_engine

# Access dashboard in your browser:
# https://localhost:9001
# https://<server-ip>:9001
```

---

## 📁 Directory Structure

```
ABLBL_AttendanceV1/
├── docker-compose.yml              # PostgreSQL + Recognition Engine configuration
├── Dockerfile                      # CUDA 12.2 / Ubuntu 22.04 server container
├── requirements.txt                # Python backend dependencies
├── package.json                    # Root build scripts
├── startup_commands.txt            # Quick reference guide & operational commands
│
├── config/
│   ├── config.yaml                 # System thresholds, models, DB, camera config
│   ├── ssl.crt                     # SSL certificate
│   └── ssl.key                     # SSL private key
│
├── frontend/                       # React 18 + Vite Web Application
│   ├── src/
│   │   ├── App.jsx                 # State orchestrator & polling loop
│   │   ├── components/             # Header, VideoPlayer, ActivityFeed, Modals
│   │   ├── engine/
│   │   │   ├── UltraLightDetector.js  # Client ONNX face detector (WASM/WebGL)
│   │   │   ├── ClientIoUTracker.js    # IoU + Proximity tracker w/ jitter filter
│   │   │   ├── CropDispatcher.js      # Gate, settle, 5-frame sampling logic
│   │   │   └── CanvasRenderer.js      # HUD, bounding boxes, ID card popup
│   │   └── services/api.js         # REST client
│   └── vite.config.js
│
├── models/
│   ├── kprpe_adaface.onnx          # AdaFace 512-d recognition model
│   ├── scrfd_2.5g_kps.onnx         # SCRFD 5-landmark face detector
│   └── ultra_light/
│       └── version-RFB-320.onnx    # Browser client 1.3MB face detector
│
├── src/                            # Python Backend Server
│   ├── api/
│   │   ├── app.py                  # FastAPI app factory & static routing
│   │   └── routes.py               # REST endpoints & photo delivery
│   ├── pipeline.py                 # Core recognition pipeline & state machine
│   ├── detection/scrfd_detector.py # SCRFD ONNX landmark detector
│   ├── recognition/kprpe_adaface.py# AdaFace GPU embedding extractor
│   ├── search/faiss_index.py       # FAISS vector gallery (IndexFlatIP)
│   ├── attendance/deduplication.py # Cooldown & deduplication controller
│   ├── enrollment/enroll_service.py# Employee enrollment pipeline
│   └── database/                   # SQLAlchemy models + PostgreSQL/SQLite repo
│
├── scripts/
│   ├── enroll_employees.py         # Batch enroll from Employees_Photo/
│   ├── benchmark.py                # Pipeline latency benchmark
│   ├── build_engines.py            # TensorRT engine compiler
│   └── generate_architecture_report_pdf.py # Architecture report generator
│
└── tests/
    └── test_components.py          # Component verification test suite
```

---

## 📡 REST API Reference

| Endpoint | Method | Request Payload | Description |
|---|:---:|---|---|
| `/api/status` | `GET` | — | System health, total enrolled count, present/absent counts, active mode |
| `/api/mode` | `GET`/`POST` | `{"mode": "ENTRY"\|"EXIT"}` | Switch system operating mode (clears deduplication cooldowns) |
| `/api/process_crop` | `POST` | `{"crop_base64": "..."}` or raw JPEG | Runs AdaFace GPU embedding, FAISS search, logs attendance |
| `/api/record_unknown` | `POST` | `{"crop_base64": "..."}` | Logs an unverified person after 5 failed sampling attempts |
| `/api/attendance/recent` | `GET` | `?limit=50` | Returns recent attendance events with snapshot and portrait URLs |
| `/api/attendance/flush` | `POST` | — | Clears all attendance records from DB and resets presence status |
| `/api/employees` | `GET` | — | Directory of all enrolled employees with real-time presence indicators |
| `/api/enroll` | `POST` | Multipart form (`photo`, `name`, `employee_id`, `department`) | Enrolls new employee, indexes face embedding, updates DB & caches |
| `/api/employees/{id}` | `DELETE`| — | **Completely purges employee from FAISS, DB, memory, and disk** |
| `/photos/{filename}` | `GET` | — | Serves enrolled employee reference portrait |
| `/captures/{filename}` | `GET` | — | Serves captured attendance snapshot image |

---

## 🔧 Production Configuration (`config/config.yaml`)

```yaml
hardware:
  device: "cuda"          # "cuda" or "cpu"
  precision: "fp16"       # fp16 for speed, fp32 for accuracy

recognition:
  match_threshold: 0.53   # Cosine similarity threshold (baseline >= 0.53)
  margin_threshold: 0.04  # Delta gate between Top-1 and Top-2 match
  unknown_threshold: 0.35 # Below this threshold = UNKNOWN

attendance:
  cooldown_seconds: 30    # Seconds before a repeat check-in is logged again
  re_entry_cooldown: 30   # Minimum interval for RE_ENTRY logging

database:
  url: "postgresql://postgres:postgres@attendance_postgres:5432/attendance_db"
  sqlite_fallback_url: "sqlite:///data/attendance.db"
```

---

## 🛠️ Management & Operational Commands

```bash
# Rebuild React frontend
npm run build

# Restart recognition engine container
docker compose restart recognition_engine

# View live container recognition logs
docker compose logs -f recognition_engine

# Batch enroll employee portraits from Employees_Photo/
# (Format: "Firstname Lastname EMP_ID.jpg")
PYTHONPATH=. python3 scripts/enroll_employees.py

# Run component unit tests
pytest tests/test_components.py -v

# Run performance benchmark
PYTHONPATH=. python3 scripts/benchmark.py

# Clear all attendance records (flush daily feed)
curl -X POST https://localhost:9001/api/attendance/flush -k

# Delete an employee completely
curl -X DELETE https://localhost:9001/api/employees/ABL188 -k
```

---

## 🛡️ License & Credits

Built for enterprise edge security and attendance monitoring by **AI Monk**.  
Proprietary and confidential — © 2026 ABLBL / AI Monk. All rights reserved.
