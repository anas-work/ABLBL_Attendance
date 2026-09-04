#!/usr/bin/env python3
"""
AI Monk Attendance System - High-Performance Edge Client for EC2 (4 GB RAM)
===========================================================================
100% 1-to-1 parity with the Edge Web App (Linzaer Ultra-Light 1MB RFB-320 ONNX,
Generalized IoU+Proximity Tracker, 120px gate, 3-second evaluation window,
and asynchronous GPU Cloud Recognition Dispatcher).

Includes an optional built-in MJPEG web streamer on port 8080 so you can view
the live stream in your browser directly from the EC2 instance!

Usage:
    # 1. Headless processing (saves annotated video)
    python3 edge_client.py --video phone_video.mp4 --server https://49.206.228.75:9001 --output annotated.mp4

    # 2. With live browser preview on http://<ec2-ip>:8080
    python3 edge_client.py --video phone_video.mp4 --server https://49.206.228.75:9001 --port 8080
"""

import os
import sys
import time
import math
import base64
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
import onnxruntime as ort
import requests
import urllib3

# Suppress self-signed certificate warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ==============================================================================
# 1. ULTRA-LIGHT RFB-320 ONNX DETECTOR (Exact match with UltraLightDetector.js)
# ==============================================================================
class UltraLightDetector:
    def __init__(self, model_path: str = "models/ultra_light/version-RFB-320.onnx", conf_threshold: float = 0.58, nms_threshold: float = 0.25):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.width = 320
        self.height = 240

        if not os.path.exists(model_path):
            candidates = [
                model_path,
                "version-RFB-320.onnx",
                "models/ultra_light/version-RFB-320.onnx",
                os.path.join(os.path.dirname(__file__), "../models/ultra_light/version-RFB-320.onnx"),
                os.path.join(os.path.dirname(__file__), "version-RFB-320.onnx")
            ]
            for c in candidates:
                if os.path.exists(c):
                    model_path = c
                    break

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.priors = self._generate_priors()

    def _generate_priors(self) -> np.ndarray:
        min_boxes = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
        strides = [8, 16, 32, 64]
        priors = []
        for i, stride in enumerate(strides):
            feat_h = math.ceil(self.height / stride)
            feat_w = math.ceil(self.width / stride)
            for h in range(feat_h):
                for w in range(feat_w):
                    for min_box in min_boxes[i]:
                        s_kx = min_box / self.width
                        s_ky = min_box / self.height
                        dense_cx = (w + 0.5) * stride / self.width
                        dense_cy = (h + 0.5) * stride / self.height
                        priors.append([dense_cx, dense_cy, s_kx, s_ky])
        return np.array(priors, dtype=np.float32)

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        orig_h, orig_w = frame.shape[:2]
        resized = cv2.resize(frame, (self.width, self.height))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = ((rgb.astype(np.float32) - 127.0) / 128.0).transpose(2, 0, 1)
        tensor = np.expand_dims(tensor, axis=0)

        outputs = self.session.run(None, {self.input_name: tensor})
        scores, boxes = outputs[0][0], outputs[1][0]  # scores: (4420, 2), boxes: (4420, 4)

        detections = []
        for i in range(len(self.priors)):
            score = float(scores[i, 1])
            if score >= self.conf_threshold:
                prior = self.priors[i]
                box = boxes[i]

                cx = prior[0] + box[0] * 0.1 * prior[2]
                cy = prior[1] + box[1] * 0.1 * prior[3]
                w = prior[2] * math.exp(box[2] * 0.2)
                h = prior[3] * math.exp(box[3] * 0.2)

                x1 = max(0.0, (cx - w / 2.0) * orig_w)
                y1 = max(0.0, (cy - h / 2.0) * orig_h)
                x2 = min(float(orig_w), (cx + w / 2.0) * orig_w)
                y2 = min(float(orig_h), (cy + h / 2.0) * orig_h)

                box_w = x2 - x1
                box_h = y2 - y1
                aspect = box_w / max(1.0, box_h)

                if box_w >= 32 and box_h >= 32 and (0.45 <= aspect <= 1.65):
                    detections.append({"bbox": [x1, y1, x2, y2], "score": score})

        return self._apply_nms(detections)

    def _apply_nms(self, dets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(dets) <= 1:
            return dets
        dets.sort(key=lambda x: x["score"], reverse=True)
        keep = []
        suppressed = [False] * len(dets)

        for i in range(len(dets)):
            if suppressed[i]:
                continue
            keep.append(dets[i])
            b1 = dets[i]["bbox"]
            for j in range(i + 1, len(dets)):
                if suppressed[j]:
                    continue
                b2 = dets[j]["bbox"]
                xx1 = max(b1[0], b2[0])
                yy1 = max(b1[1], b2[1])
                xx2 = min(b1[2], b2[2])
                yy2 = min(b1[3], b2[3])
                w = max(0.0, xx2 - xx1 + 1)
                h = max(0.0, yy2 - yy1 + 1)
                inter = w * h
                area1 = (b1[2] - b1[0] + 1) * (b1[3] - b1[1] + 1)
                area2 = (b2[2] - b2[0] + 1) * (b2[3] - b2[1] + 1)
                iou = inter / max(1.0, area1 + area2 - inter)
                if iou >= self.nms_threshold:
                    suppressed[j] = True
        return keep


# ==============================================================================
# 2. CLIENT IoU + PROXIMITY MOTION TRACKER (Exact match with ClientIoUTracker.js)
# ==============================================================================
class EdgeIoUTracker:
    def __init__(self, iou_threshold: float = 0.12, max_age: int = 45):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.next_id = 101
        self.tracks = []

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # 1. Motion interpolation
        for t in self.tracks:
            t["age"] += 1
            t["time_since_update"] += 1
            if 0 < t["time_since_update"] <= 8:
                vx, vy = t.get("vx", 0.0) * 0.8, t.get("vy", 0.0) * 0.8
                t["bbox"] = [t["bbox"][0] + vx, t["bbox"][1] + vy, t["bbox"][2] + vx, t["bbox"][3] + vy]

        if not detections:
            self.tracks = [t for t in self.tracks if t["time_since_update"] <= self.max_age]
            return [t for t in self.tracks if t["time_since_update"] <= 15]

        if not self.tracks:
            for d in detections:
                self.tracks.append({
                    "track_id": self.next_id,
                    "bbox": list(d["bbox"]),
                    "score": d["score"],
                    "age": 1,
                    "hits": 1,
                    "time_since_update": 0,
                    "vx": 0.0,
                    "vy": 0.0,
                    "recognition_state": "WAITING_FOR_SIZE",
                    "assigned_identity": None,
                    "employee_id": None,
                    "confidence": 0.0,
                    "decision": "UNKNOWN",
                    "photo_url": None,
                    "first_seen_at_120px": None,
                    "last_attempt_time": 0.0,
                    "in_flight": False
                })
                self.next_id += 1
            return self.tracks

        # 2. Generalized Association Matrix (IoU * 0.65 + Proximity * 0.35)
        iou_matrix = []
        for t in self.tracks:
            row = []
            tb = t["bbox"]
            tcx, tcy = (tb[0] + tb[2]) / 2.0, (tb[1] + tb[3]) / 2.0
            t_size = max(tb[2] - tb[0], tb[3] - tb[1])

            for d in detections:
                db = d["bbox"]
                dcx, dcy = (db[0] + db[2]) / 2.0, (db[1] + db[3]) / 2.0
                d_size = max(db[2] - db[0], db[3] - db[1])

                iou = self._compute_iou(tb, db)
                dist = math.hypot(tcx - dcx, tcy - dcy)
                avg_size = max(30.0, (t_size + d_size) / 2.0)
                prox = max(0.0, 1.0 - (dist / (avg_size * 1.5)))

                score = iou * 0.65 + prox * 0.35
                row.append(score)
            iou_matrix.append(row)

        matched_tracks = set()
        matched_dets = set()

        while True:
            max_iou = -1.0
            max_t = -1
            max_d = -1

            for t_idx in range(len(self.tracks)):
                if t_idx in matched_tracks:
                    continue
                for d_idx in range(len(detections)):
                    if d_idx in matched_dets:
                        continue
                    if iou_matrix[t_idx][d_idx] > max_iou:
                        max_iou = iou_matrix[t_idx][d_idx]
                        max_t = t_idx
                        max_d = d_idx

            if max_iou < self.iou_threshold or max_t == -1:
                break

            matched_tracks.add(max_t)
            matched_dets.add(max_d)

            track = self.tracks[max_t]
            det = detections[max_d]
            tb = track["bbox"]
            db = det["bbox"]

            tcx, tcy = (tb[0] + tb[2]) / 2.0, (tb[1] + tb[3]) / 2.0
            dcx, dcy = (db[0] + db[2]) / 2.0, (db[1] + db[3]) / 2.0

            inst_vx = (dcx - tcx) * 0.35
            inst_vy = (dcy - tcy) * 0.35
            track["vx"] = track.get("vx", 0.0) * 0.7 + inst_vx * 0.3
            track["vy"] = track.get("vy", 0.0) * 0.7 + inst_vy * 0.3

            # Coordinate smoothing EMA (75% det, 25% track)
            track["bbox"] = [
                0.75 * db[0] + 0.25 * tb[0],
                0.75 * db[1] + 0.25 * tb[1],
                0.75 * db[2] + 0.25 * tb[2],
                0.75 * db[3] + 0.25 * tb[3]
            ]
            track["score"] = det["score"]
            track["hits"] += 1
            track["time_since_update"] = 0

        # Unmatched detections -> new tracks
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_dets:
                self.tracks.append({
                    "track_id": self.next_id,
                    "bbox": list(det["bbox"]),
                    "score": det["score"],
                    "age": 1,
                    "hits": 1,
                    "time_since_update": 0,
                    "vx": 0.0,
                    "vy": 0.0,
                    "recognition_state": "WAITING_FOR_SIZE",
                    "assigned_identity": None,
                    "employee_id": None,
                    "confidence": 0.0,
                    "decision": "UNKNOWN",
                    "photo_url": None,
                    "first_seen_at_120px": None,
                    "last_attempt_time": 0.0,
                    "in_flight": False
                })
                self.next_id += 1

        # 3. Inter-track containment & proximity deduplication
        valid_tracks = []
        for i in range(len(self.tracks)):
            tA = self.tracks[i]
            if tA["time_since_update"] > self.max_age:
                continue
            is_dup = False
            for j in range(len(valid_tracks)):
                tB = valid_tracks[j]
                iou = self._compute_iou(tA["bbox"], tB["bbox"])
                iom = self._compute_iom(tA["bbox"], tB["bbox"])
                cAx = (tA["bbox"][0] + tA["bbox"][2]) / 2.0
                cAy = (tA["bbox"][1] + tA["bbox"][3]) / 2.0
                cBx = (tB["bbox"][0] + tB["bbox"][2]) / 2.0
                cBy = (tB["bbox"][1] + tB["bbox"][3]) / 2.0
                center_dist = math.hypot(cAx - cBx, cAy - cBy)
                avg_size = ((tA["bbox"][2] - tA["bbox"][0]) + (tB["bbox"][2] - tB["bbox"][0])) / 2.0

                if iou >= 0.15 or iom >= 0.25 or center_dist < avg_size * 0.75:
                    if tA["recognition_state"] == "MATCHED" and tB["recognition_state"] != "MATCHED":
                        tB["recognition_state"] = "MATCHED"
                        tB["assigned_identity"] = tA["assigned_identity"]
                        tB["employee_id"] = tA["employee_id"]
                        tB["confidence"] = tA["confidence"]
                        tB["photo_url"] = tA["photo_url"]
                        tB["decision"] = tA["decision"]
                    is_dup = True
                    break
            if not is_dup:
                valid_tracks.append(tA)

        self.tracks = valid_tracks
        return [t for t in self.tracks if t["time_since_update"] <= 15]

    def _compute_iou(self, boxA, boxB) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0.0, xB - xA + 1) * max(0.0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        return interArea / max(1.0, (boxAArea + boxBArea - interArea))

    def _compute_iom(self, boxA, boxB) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
        areaA = max(1.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        areaB = max(1.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
        return interArea / min(areaA, areaB)


# ==============================================================================
# 3. EDGE RECOGNITION DISPATCHER (Exact match with CropDispatcher.js)
# ==============================================================================
class EdgeDispatcher:
    def __init__(self, server_url: str, min_size: int = 120, unknown_timeout: float = 3.0):
        self.server_url = server_url.rstrip("/")
        self.min_size = min_size
        self.unknown_timeout = unknown_timeout
        self.pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="edge_dispatch")
        self.session = requests.Session()
        self.session.verify = False
        self.active_popup: Optional[Dict[str, Any]] = None

    def process_tracks(self, frame: np.ndarray, tracks: List[Dict[str, Any]], now: float, mode: str):
        h, w = frame.shape[:2]
        
        # Sort tracks by area descending
        cand_tracks = sorted(
            [t for t in tracks if t["time_since_update"] <= 10],
            key=lambda t: (t["bbox"][2] - t["bbox"][0]) * (t["bbox"][3] - t["bbox"][1]),
            reverse=True
        )

        for i, track in enumerate(cand_tracks):
            bx1, by1, bx2, by2 = track["bbox"]
            face_w = bx2 - bx1
            face_h = by2 - by1
            face_size = max(face_w, face_h)

            if face_size < self.min_size:
                track["first_seen_at_120px"] = None
                if track["recognition_state"] not in ["MATCHED", "NOT_RECOGNIZED"]:
                    track["recognition_state"] = "WAITING_FOR_SIZE"
                continue

            if not track.get("first_seen_at_120px"):
                track["first_seen_at_120px"] = now

            if track["recognition_state"] == "MATCHED":
                continue

            if face_size < self.min_size:
                track["first_seen_at_120px"] = None
                if track["recognition_state"] != "NOT_RECOGNIZED":
                    track["recognition_state"] = "WAITING_FOR_SIZE"
                continue

            # Ignore cut-off border faces
            if bx1 <= 6 or by1 <= 6 or bx2 >= w - 6 or by2 >= h - 6:
                continue

            if not track.get("first_seen_at_120px"):
                track["first_seen_at_120px"] = now
                track["eval_attempts"] = 0

            if i >= 2:
                continue

            throttle_ms = 0.28 if track["recognition_state"] == "NOT_RECOGNIZED" else 0.11
            if not track.get("in_flight") and (now - track.get("last_attempt_time", 0.0) > throttle_ms):
                track["in_flight"] = True
                track["last_attempt_time"] = now
                track["eval_attempts"] = track.get("eval_attempts", 0) + 1
                if track["recognition_state"] != "NOT_RECOGNIZED":
                    track["recognition_state"] = "ANALYZING"

                # 25% margin padding
                pad_x = max(10, int(face_w * 0.25))
                pad_y = max(10, int(face_h * 0.25))
                cx1 = max(0, int(bx1 - pad_x))
                cy1 = max(0, int(by1 - pad_y))
                cx2 = min(w, int(bx2 + pad_x))
                cy2 = min(h, int(by2 + pad_y))

                crop_img = frame[cy1:cy2, cx1:cx2]
                if crop_img.size == 0:
                    track["in_flight"] = False
                    continue

                crop_resized = cv2.resize(crop_img, (224, 224))
                _, crop_enc = cv2.imencode(".jpg", crop_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                crop_b64 = "data:image/jpeg;base64," + base64.b64encode(crop_enc.tobytes()).decode("utf-8")

                _, full_enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                full_b64 = "data:image/jpeg;base64," + base64.b64encode(full_enc.tobytes()).decode("utf-8")

                self.pool.submit(self._send_crop_request, track, crop_b64, full_b64, mode, now)

    def _send_crop_request(self, track: Dict[str, Any], crop_b64: str, full_b64: str, mode: str, req_time: float):
        try:
            url = f"{self.server_url}/api/process_crop"
            resp = self.session.post(url, json={"crop_base64": crop_b64, "full_frame_base64": full_b64}, timeout=3.0)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") == "SUCCESS" and data.get("matched"):
                    track["recognition_state"] = "MATCHED"
                    track["assigned_identity"] = data.get("name")
                    track["employee_id"] = data.get("employee_id")
                    track["confidence"] = data.get("confidence", 0.0)
                    track["photo_url"] = data.get("photo_url")
                    track["decision"] = data.get("decision", "CHECK_OUT" if mode == "EXIT" else "CHECK_IN")
                    
                    self.active_popup = {
                        "name": data.get("name"),
                        "employee_id": data.get("employee_id"),
                        "confidence": data.get("confidence", 0.0),
                        "decision": track["decision"],
                        "photo_url": data.get("photo_url"),
                        "is_unknown": False,
                        "until": time.time() + 2.5
                    }
                    print(f"✅ [MATCHED] {data.get('name')} ({data.get('employee_id')}) - Score: {data.get('confidence'):.2%}")
                else:
                    # Unmatched attempt: check if 3.5s window AND at least 5 attempts have elapsed
                    elapsed = time.time() - (track.get("first_seen_at_120px") or req_time)
                    if elapsed >= self.unknown_timeout and track.get("eval_attempts", 0) >= 5:
                        if not track.get("unknown_recorded"):
                            track["unknown_recorded"] = True
                            track["recognition_state"] = "NOT_RECOGNIZED"
                            track["assigned_identity"] = "Unknown Person"
                            track["employee_id"] = "UNKNOWN"
                            track["decision"] = "UNKNOWN"
                            track["confidence"] = data.get("confidence", 0.0)
                            
                            self.active_popup = {
                                "name": "Unknown Person",
                                "employee_id": "UNKNOWN",
                                "confidence": data.get("confidence", 0.0),
                                "decision": "UNKNOWN",
                                "photo_url": None,
                                "is_unknown": True,
                                "until": time.time() + 3.5
                            }
                            print(f"⚠️ [CRITICAL] Unknown Person Detected after evaluation (Track #{track['track_id']})")
                            self._record_unknown(crop_b64, full_b64)
                    else:
                        if track["recognition_state"] != "NOT_RECOGNIZED":
                            track["recognition_state"] = "ANALYZING"
        except Exception as e:
            print(f"❌ [Network Error] Could not reach GPU server at {self.server_url}: {e}")
        finally:
            track["in_flight"] = False

    def _record_unknown(self, crop_b64: str, full_b64: str):
        try:
            url = f"{self.server_url}/api/record_unknown"
            self.session.post(url, json={"crop_base64": crop_b64, "full_frame_base64": full_b64}, timeout=3.0)
        except Exception:
            pass


# ==============================================================================
# 4. OVERLAY RENDERER (Exact match with CanvasRenderer.js)
# ==============================================================================
class EdgeRenderer:
    def __init__(self, server_url: str):
        self.server_url = server_url.rstrip("/")
        self.photo_cache: Dict[str, np.ndarray] = {}

    def render(self, frame: np.ndarray, tracks: List[Dict[str, Any]], mode: str, telemetry: Dict[str, float], active_popup: Optional[Dict[str, Any]], now: float) -> np.ndarray:
        h, w = frame.shape[:2]
        active_tracks = [t for t in tracks if t["time_since_update"] <= 15]

        # 1. Bounding Boxes
        for t in active_tracks:
            bx1, by1, bx2, by2 = [int(v) for v in t["bbox"]]
            state = t["recognition_state"]

            if state == "MATCHED":
                if t["decision"] == "CHECK_OUT":
                    box_color = (239, 70, 217)  # Purple
                    label = f"CHECK-OUT: {t['assigned_identity']} [{int(t['confidence']*100)}%]"
                elif t["decision"] == "RE_ENTRY":
                    box_color = (11, 158, 245)  # Orange
                    label = f"RE-ENTRY: {t['assigned_identity']} [{int(t['confidence']*100)}%]"
                else:
                    box_color = (129, 185, 16)  # Green
                    label = f"{t['assigned_identity']} [{int(t['confidence']*100)}%]"
            elif state == "NOT_RECOGNIZED":
                box_color = (68, 68, 239)      # Red
                label = "NOT RECOGNIZED"
            elif state == "WAITING_FOR_SIZE":
                box_color = (139, 116, 100)    # Gray
                label = "APPROACH CAMERA"
            elif state == "ANALYZING":
                box_color = (212, 182, 6)       # Cyan
                label = "ANALYZING..."
            else:
                box_color = (248, 189, 56)      # Sky Blue
                label = "DETECTING..."

            thickness = 3 if state == "NOT_RECOGNIZED" else 2
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), box_color, thickness)
            
            # Label Banner
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (bx1, max(0, by1 - 24)), (bx1 + tw + 12, by1), box_color, -1)
            cv2.putText(frame, label, (bx1 + 6, max(14, by1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # 2. Top-Left Diagnostics HUD Box
        hud_w, hud_h = 320, 75
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (10 + hud_w, 10 + hud_h), (29, 15, 10), -1)
        frame = cv2.addWeighted(overlay, 0.88, frame, 0.12, 0)
        hud_border = (239, 70, 217) if mode == "EXIT" else (129, 185, 16)
        cv2.rectangle(frame, (10, 10), (10 + hud_w, 10 + hud_h), hud_border, 1)

        cv2.putText(frame, f"MODE: {mode} | FPS: {telemetry.get('fps', 30.0):.1f} | Tracks: {len(active_tracks)}", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Detect: {telemetry.get('detect_ms', 8.0):.1f}ms | Track: {telemetry.get('track_ms', 0.1):.1f}ms", (18, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"End-to-End Latency: {telemetry.get('e2e_ms', 8.2):.1f}ms", (18, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        # 3. Flashing Official ID Card HUD (Top Right)
        if active_popup and now < active_popup.get("until", 0.0):
            card_w, card_h = 360, 130
            card_x = w - card_w - 20
            card_y = 20

            is_unknown = active_popup.get("is_unknown")
            card_border = (68, 68, 239) if is_unknown else ((239, 70, 217) if active_popup.get("decision") == "CHECK_OUT" else (129, 185, 16))
            card_bg = (10, 10, 45) if is_unknown else (42, 23, 15)

            overlay = frame.copy()
            cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), card_bg, -1)
            frame = cv2.addWeighted(overlay, 0.92, frame, 0.08, 0)
            cv2.rectangle(frame, (card_x, card_y), (card_x + card_w, card_y + card_h), card_border, 3)

            # Photo Box
            px, py, pw, ph = card_x + 12, card_y + 12, 90, 105
            cv2.rectangle(frame, (px, py), (px + pw, py + ph), (0, 0, 0), -1)
            cv2.rectangle(frame, (px, py), (px + pw, py + ph), card_border, 1)

            # Text Info
            tx = card_x + 115
            title = "CRITICAL: UNKNOWN PERSON DETECTED" if is_unknown else "OFFICIAL ID CARD VERIFIED"
            title_color = (113, 113, 248) if is_unknown else (248, 189, 56)
            cv2.putText(frame, title, (tx, card_y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.38, title_color, 1, cv2.LINE_AA)

            name_str = f"IDENTITY: {active_popup.get('name', 'Unknown Person')}"
            cv2.putText(frame, name_str, (tx, card_y + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            status_str = "STATUS: UNREGISTERED / NOT RECOGNIZED" if is_unknown else f"STATUS: {active_popup.get('decision', 'PRESENT')}"
            cv2.putText(frame, status_str, (tx, card_y + 74), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (203, 213, 225), 1, cv2.LINE_AA)

            conf_str = "VERIFICATION: MATCH NOT FOUND" if is_unknown else f"CONFIDENCE: {int(active_popup.get('confidence', 0.0)*100)}%"
            conf_color = (68, 68, 239) if is_unknown else (153, 211, 52)
            cv2.putText(frame, conf_str, (tx, card_y + 96), cv2.FONT_HERSHEY_SIMPLEX, 0.40, conf_color, 1, cv2.LINE_AA)

        return frame


# ==============================================================================
# 5. LIGHTWEIGHT MJPEG PREVIEW SERVER (PORT 8080)
# ==============================================================================
latest_mjpeg_frame = None
mjpeg_lock = threading.Lock()

class MJPEGHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global latest_mjpeg_frame
        if self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            while True:
                with mjpeg_lock:
                    if latest_mjpeg_frame is None:
                        continue
                    jpg = latest_mjpeg_frame
                self.wfile.write(b"--frame\r\n")
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpg)))
                self.end_headers()
                self.wfile.write(jpg)
                self.wfile.write(b"\r\n")
                time.sleep(0.033)
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            html = """
            <!DOCTYPE html>
            <html>
            <head><title>AI Monk Attendance - Edge Stream Preview</title>
            <style>
                body { margin:0; background:#0b0f19; display:flex; flex-direction:column; align-items:center; justify-content:center; height:100vh; font-family:sans-serif; color:#fff; }
                h2 { margin-bottom:10px; }
                img { border:2px solid #00ffff; border-radius:8px; max-width:95vw; max-height:85vh; box-shadow:0 0 20px rgba(0,255,255,0.3); }
            </style>
            </head>
            <body>
                <h2>🚀 AI Monk Edge Client Live Stream</h2>
                <img src="/stream" />
            </body>
            </html>
            """
            self.wfile.write(html.encode("utf-8"))

    def log_message(self, format, *args):
        pass


def start_mjpeg_server(port: int):
    server = HTTPServer(("0.0.0.0", port), MJPEGHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"📡 Live Web Stream active at: http://0.0.0.0:{port}")


# ==============================================================================
# 6. MAIN EXECUTION LOOP
# ==============================================================================
def main():
    global latest_mjpeg_frame
    parser = argparse.ArgumentParser(description="AI Monk Attendance - Standalone Edge Client for EC2")
    parser.add_argument("--video", type=str, default="0", help="Webcam index (0) or path to video file/RTSP URL")
    parser.add_argument("--server", type=str, default="https://49.206.228.75:9001", help="Remote GPU Recognition Server URL")
    parser.add_argument("--model", type=str, default="models/ultra_light/version-RFB-320.onnx", help="Path to RFB-320 ONNX model")
    parser.add_argument("--mode", type=str, default="ENTRY", choices=["ENTRY", "EXIT"], help="Operation mode (ENTRY or EXIT)")
    parser.add_argument("--output", type=str, default="", help="Optional path to save annotated output video (e.g. output.mp4)")
    parser.add_argument("--loop", action="store_true", default=True, help="Loop video continuously for live stream")
    parser.add_argument("--no-loop", dest="loop", action="store_false", help="Do not loop video")
    parser.add_argument("--realtime", action="store_true", default=True, help="Pace playback at natural video FPS (e.g. 24/30 FPS)")
    parser.add_argument("--fast", dest="realtime", action="store_false", help="Process at maximum CPU speed without FPS throttling")
    parser.add_argument("--port", type=int, default=0, help="Optional port to launch live MJPEG preview web stream (e.g. 8080)")
    parser.add_argument("--show", action="store_true", help="Display local OpenCV preview window (requires GUI/X11)")
    args = parser.parse_args()

    # Verify input
    video_src = int(args.video) if args.video.isdigit() else args.video
    cap = cv2.VideoCapture(video_src)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video source '{args.video}'")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    target_frame_time = 1.0 / max(1.0, min(60.0, fps))

    print("=" * 75)
    print("🚀 AI MONK ATTENDANCE - EDGE CLIENT (EC2 4GB RAM)")
    print(f"📹 Video Source:   {args.video} ({width}x{height} @ {fps:.1f} FPS)")
    print(f"🌐 GPU Server:     {args.server}")
    print(f"🧠 Edge Model:     {args.model} (Linzaer Ultra-Light 1MB RFB-320 ONNX)")
    print(f"⚙️  Operation Mode: {args.mode}")
    print(f"🔄 Loop Mode:      {'ENABLED (Continuous live stream)' if args.loop else 'DISABLED (Process once and stop)'}")
    print(f"⏱️  Pacing:         {'Real-time ' + str(round(fps, 1)) + ' FPS' if args.realtime else 'Maximum speed (unthrottled)'}")
    print("=" * 75)

    detector = UltraLightDetector(model_path=args.model)
    tracker = EdgeIoUTracker()
    dispatcher = EdgeDispatcher(server_url=args.server)
    renderer = EdgeRenderer(server_url=args.server)

    if args.port > 0:
        start_mjpeg_server(args.port)

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"💾 Saving annotated video to: {args.output}")

    frame_idx = 0
    t0_fps = time.time()
    telemetry = {"fps": fps, "detect_ms": 8.0, "track_ms": 0.1, "e2e_ms": 8.2}

    try:
        while True:
            t_frame_start = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                if args.loop and isinstance(video_src, str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.1)
                    continue
                else:
                    print("🏁 End of video reached.")
                    break

            now = time.time()
            frame_idx += 1

            # 1. Edge Detection
            t_det_s = time.perf_counter()
            dets = detector.detect(frame)
            telemetry["detect_ms"] = (time.perf_counter() - t_det_s) * 1000.0

            # 2. Motion Tracking
            t_trk_s = time.perf_counter()
            active_tracks = tracker.update(dets)
            telemetry["track_ms"] = (time.perf_counter() - t_trk_s) * 1000.0

            # 3. Recognition Dispatching (120px gate & 3s window)
            dispatcher.process_tracks(frame, active_tracks, now, args.mode)

            # 4. Measure FPS
            if now - t0_fps >= 1.0:
                telemetry["fps"] = frame_idx / (now - t0_fps)
                frame_idx = 0
                t0_fps = now

            telemetry["e2e_ms"] = (time.perf_counter() - t_frame_start) * 1000.0

            # 5. Render Annotations & HUD
            annotated_frame = renderer.render(frame, active_tracks, args.mode, telemetry, dispatcher.active_popup, now)

            if writer:
                writer.write(annotated_frame)

            if args.port > 0:
                _, jpg_bytes = cv2.imencode(".jpg", annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                with mjpeg_lock:
                    latest_mjpeg_frame = jpg_bytes.tobytes()

            if args.show:
                cv2.imshow("AI Monk Attendance Edge", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # 6. Real-time FPS pacing so video plays at natural speed
            if args.realtime:
                proc_time = time.perf_counter() - t_frame_start
                sleep_sec = target_frame_time - proc_time
                if sleep_sec > 0.002:
                    time.sleep(sleep_sec)

    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
    finally:
        cap.release()
        if writer:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()
        print("✅ Edge Client cleanly stopped.")


if __name__ == "__main__":
    main()
