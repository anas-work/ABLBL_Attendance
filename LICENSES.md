# Software & Model Licensing Compliance Matrix

This document provides a breakdown of all third-party software, deep learning models, pretrained checkpoints, and library dependencies used in the Real-Time Employee Face Recognition and Attendance System.

---

## 1. Deep Learning Models & Pretrained Checkpoints

| Component | Repository / Source | Code License | Pretrained Weights License | Commercial Deployment Note |
| :--- | :--- | :--- | :--- | :--- |
| **SCRFD Detector** | InsightFace (`deepinsight/insightface`) | Apache-2.0 | InsightFace Non-Commercial Research License | Commercial deployment requires explicit license from InsightFace / DeepInsight if using official pretrained weights. Custom trained SCRFD models on open commercial datasets are fully permitted. |
| **AdaFace Recognizer** | AdaFace (`minchulkim/AdaFace`) | MIT License | MIT / Research License (depends on training dataset like MS1MV2 or WebFace) | Code is MIT. MS1MV2/WebFace training datasets have research-only clauses; for commercial deployments, train AdaFace on commercially licensed datasets (e.g. BUPT-Balanced). |
| **KPRPE Architecture** | KPRPE (`KeyPoint Relative Position Encoding`) | Apache-2.0 / MIT | Research Checkpoint | KeyPoint Relative Position Encoding logic is open-source. Pretrained weights carry underlying dataset restrictions. |

---

## 2. Infrastructure & Library Dependencies

| Library | License | Usage Purpose |
| :--- | :--- | :--- |
| **FAISS** | MIT License (Meta AI) | Fast local vector search & similarity indexing |
| **NVIDIA TensorRT** | NVIDIA Software License Agreement | Hardware acceleration engines on RTX server & Jetson edge |
| **ONNX Runtime GPU** | MIT License (Microsoft) | Cross-platform deep learning execution engine |
| **PyTorch & Torchvision**| BSD-style License | PyTorch model conversion & GPU tensor utilities |
| **OpenCV** | Apache-2.0 | Computer vision, video stream decoding, and image matrix processing |
| **FastAPI** | MIT License | Web API framework and dashboard backend |
| **PostgreSQL & SQLAlchemy**| PostgreSQL License / MIT | Authoritative attendance & employee relational database |

---

## 3. Biometric & Data Privacy Statement

All employee photographs, facial keypoint metadata, normalized 512-dimensional embeddings, and attendance event logs remain **100% strictly stored on-premise** on local servers or NVIDIA Jetson edge devices. No biometric features, images, or telemetry are transmitted to cloud APIs or third-party external networks.
