#!/usr/bin/env python3
"""
AI Monk Attendance System - Edge Client Benchmarking Suite
=========================================================
Profiles system resource utilization, memory footprint (RSS/VMS), CPU load,
detection/tracking latency, cloud recognition throughput, and real-time processing FPS.

Usage:
    python3 benchmark_edge.py --video phone_video.mp4 --server https://49.206.228.75:9001
"""

import os
import sys
import time
import math
import base64
import argparse
import platform
import subprocess
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
import onnxruntime as ort
import requests
import urllib3

# Suppress certificate warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


# ==============================================================================
# 1. HARDWARE TELEMETRY PROFILER
# ==============================================================================
def get_hardware_info() -> Dict[str, Any]:
    info = {
        "os": f"{platform.system()} {platform.release()}",
        "architecture": platform.machine(),
        "cpu_model": "Unknown CPU",
        "cpu_cores_logical": os.cpu_count() or 1,
        "total_ram_gb": 0.0,
        "available_ram_gb": 0.0
    }

    # Get CPU Model from /proc/cpuinfo
    if os.path.exists("/proc/cpuinfo"):
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        info["cpu_model"] = line.split(":", 1)[1].strip()
                        break
        except Exception:
            pass

    # Get RAM from psutil or /proc/meminfo
    if HAS_PSUTIL:
        mem = psutil.virtual_memory()
        info["total_ram_gb"] = round(mem.total / (1024 ** 3), 2)
        info["available_ram_gb"] = round(mem.available / (1024 ** 3), 2)
    elif os.path.exists("/proc/meminfo"):
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if "MemTotal:" in line:
                        total_kb = int(line.split()[1])
                        info["total_ram_gb"] = round(total_kb / (1024 ** 2), 2)
                    elif "MemAvailable:" in line:
                        avail_kb = int(line.split()[1])
                        info["available_ram_gb"] = round(avail_kb / (1024 ** 2), 2)
        except Exception:
            pass

    return info


def get_process_memory_mb() -> float:
    if HAS_PSUTIL:
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    elif os.path.exists("/proc/self/status"):
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return float(line.split()[1]) / 1024.0
        except Exception:
            pass
    return 0.0


# ==============================================================================
# 2. ULTRA-LIGHT RFB-320 ONNX DETECTOR
# ==============================================================================
class BenchmarkDetector:
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
        scores, boxes = outputs[0][0], outputs[1][0]

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

        if len(detections) <= 1:
            return detections

        detections.sort(key=lambda x: x["score"], reverse=True)
        keep = []
        suppressed = [False] * len(detections)

        for i in range(len(detections)):
            if suppressed[i]:
                continue
            keep.append(detections[i])
            b1 = detections[i]["bbox"]
            for j in range(i + 1, len(detections)):
                if suppressed[j]:
                    continue
                b2 = detections[j]["bbox"]
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
# 3. BENCHMARK IoU TRACKER
# ==============================================================================
class BenchmarkTracker:
    def __init__(self, iou_threshold: float = 0.12, max_age: int = 45):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.next_id = 101
        self.tracks = []

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
                    "first_seen_at_120px": None,
                    "last_attempt_time": 0.0,
                    "in_flight": False
                })
                self.next_id += 1
            return self.tracks

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

                xx1 = max(tb[0], db[0])
                yy1 = max(tb[1], db[1])
                xx2 = min(tb[2], db[2])
                yy2 = min(tb[3], db[3])
                inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
                area1 = (tb[2] - tb[0]) * (tb[3] - tb[1])
                area2 = (db[2] - db[0]) * (db[3] - db[1])
                iou = inter / max(1.0, area1 + area2 - inter)

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
            max_t, max_d = -1, -1
            for t_idx in range(len(self.tracks)):
                if t_idx in matched_tracks:
                    continue
                for d_idx in range(len(detections)):
                    if d_idx in matched_dets:
                        continue
                    if iou_matrix[t_idx][d_idx] > max_iou:
                        max_iou = iou_matrix[t_idx][d_idx]
                        max_t, max_d = t_idx, d_idx

            if max_iou < self.iou_threshold or max_t == -1:
                break

            matched_tracks.add(max_t)
            matched_dets.add(max_d)

            track = self.tracks[max_t]
            det = detections[max_d]
            tb, db = track["bbox"], det["bbox"]
            tcx, tcy = (tb[0] + tb[2]) / 2.0, (tb[1] + tb[3]) / 2.0
            dcx, dcy = (db[0] + db[2]) / 2.0, (db[1] + db[3]) / 2.0

            inst_vx = (dcx - tcx) * 0.35
            inst_vy = (dcy - tcy) * 0.35
            track["vx"] = track.get("vx", 0.0) * 0.7 + inst_vx * 0.3
            track["vy"] = track.get("vy", 0.0) * 0.7 + inst_vy * 0.3

            track["bbox"] = [
                0.75 * db[0] + 0.25 * tb[0],
                0.75 * db[1] + 0.25 * tb[1],
                0.75 * db[2] + 0.25 * tb[2],
                0.75 * db[3] + 0.25 * tb[3]
            ]
            track["score"] = det["score"]
            track["hits"] += 1
            track["time_since_update"] = 0

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
                    "first_seen_at_120px": None,
                    "last_attempt_time": 0.0,
                    "in_flight": False
                })
                self.next_id += 1

        self.tracks = [t for t in self.tracks if t["time_since_update"] <= self.max_age]
        return [t for t in self.tracks if t["time_since_update"] <= 15]


# ==============================================================================
# 4. BENCHMARK EXECUTION ENGINE
# ==============================================================================
def run_benchmark(video_path: str, server_url: str, model_path: str, max_frames: int = 0) -> Dict[str, Any]:
    hw_info = get_hardware_info()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video file '{video_path}'")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    video_duration_sec = total_video_frames / max(1.0, native_fps)

    detector = BenchmarkDetector(model_path=model_path)
    tracker = BenchmarkTracker()
    session = requests.Session()
    session.verify = False

    print("=" * 80)
    print("⚡ AI MONK EDGE CLIENT PERFORMANCE & RESOURCE BENCHMARK")
    print("=" * 80)
    print(f"🖥️  CPU:               {hw_info['cpu_model']} ({hw_info['cpu_cores_logical']} Logical Cores)")
    print(f"🧠 Total System RAM:  {hw_info['total_ram_gb']} GB (Available: {hw_info['available_ram_gb']} GB)")
    print(f"📹 Video File:        {video_path} ({width}x{height} @ {native_fps:.1f} FPS, {total_video_frames} Frames, {video_duration_sec:.1f}s)")
    print(f"🌐 Remote GPU Server: {server_url}")
    print("=" * 80)

    detect_times = []
    track_times = []
    e2e_times = []
    api_latencies = []
    peak_ram_mb = 0.0

    matched_identities = set()
    unknown_incidents = 0
    total_crop_dispatches = 0
    total_detections_count = 0

    processed_frames = 0
    t_start_total = time.perf_counter()

    try:
        while True:
            if max_frames > 0 and processed_frames >= max_frames:
                break

            t_f0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                break

            processed_frames += 1

            # 1. Detection Phase
            t_d0 = time.perf_counter()
            dets = detector.detect(frame)
            t_detect = (time.perf_counter() - t_d0) * 1000.0
            detect_times.append(t_detect)
            total_detections_count += len(dets)

            # 2. Tracking Phase
            t_t0 = time.perf_counter()
            active_tracks = tracker.update(dets)
            t_track = (time.perf_counter() - t_t0) * 1000.0
            track_times.append(t_track)

            # 3. Recognition Dispatch Phase
            now = time.time()
            for track in active_tracks:
                bx1, by1, bx2, by2 = track["bbox"]
                face_w, face_h = bx2 - bx1, by2 - by1
                face_size = max(face_w, face_h)

                if face_size >= 120 and track["recognition_state"] not in ["MATCHED", "NOT_RECOGNIZED"]:
                    if not track.get("first_seen_at_120px"):
                        track["first_seen_at_120px"] = now

                    if now - track.get("last_attempt_time", 0.0) > 0.12:
                        track["last_attempt_time"] = now
                        total_crop_dispatches += 1

                        # Extract 224x224 crop
                        pad_x, pad_y = max(10, int(face_w * 0.25)), max(10, int(face_h * 0.25))
                        cx1, cy1 = max(0, int(bx1 - pad_x)), max(0, int(by1 - pad_y))
                        cx2, cy2 = min(width, int(bx2 + pad_x)), min(height, int(by2 + pad_y))
                        crop_img = frame[cy1:cy2, cx1:cx2]

                        if crop_img.size > 0:
                            crop_resized = cv2.resize(crop_img, (224, 224))
                            _, crop_enc = cv2.imencode(".jpg", crop_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                            crop_b64 = "data:image/jpeg;base64," + base64.b64encode(crop_enc.tobytes()).decode("utf-8")

                            t_api0 = time.perf_counter()
                            try:
                                resp = session.post(
                                    f"{server_url.rstrip('/')}/api/process_crop",
                                    json={"crop_base64": crop_b64, "full_frame_base64": crop_b64},
                                    timeout=2.0
                                )
                                api_latency = (time.perf_counter() - t_api0) * 1000.0
                                api_latencies.append(api_latency)

                                if resp.status_code == 200:
                                    res_data = resp.json()
                                    if res_data.get("matched"):
                                        track["recognition_state"] = "MATCHED"
                                        track["assigned_identity"] = res_data.get("name")
                                        matched_identities.add(f"{res_data.get('name')} ({res_data.get('employee_id')})")
                                    elif (now - track["first_seen_at_120px"]) >= 3.0:
                                        track["recognition_state"] = "NOT_RECOGNIZED"
                                        unknown_incidents += 1
                            except Exception:
                                pass

            # Measure Memory
            cur_ram = get_process_memory_mb()
            if cur_ram > peak_ram_mb:
                peak_ram_mb = cur_ram

            e2e_times.append((time.perf_counter() - t_f0) * 1000.0)

            if processed_frames % 30 == 0 or processed_frames == total_video_frames:
                elapsed_now = time.perf_counter() - t_start_total
                cur_fps = processed_frames / max(0.001, elapsed_now)
                print(f"📊 Progress: {processed_frames}/{total_video_frames} Frames ({(processed_frames/total_video_frames)*100:.1f}%) | Speed: {cur_fps:.1f} FPS | RAM: {cur_ram:.1f} MB")

    finally:
        cap.release()

    total_time_sec = time.perf_counter() - t_start_total
    avg_fps = processed_frames / max(0.001, total_time_sec)
    speedup_ratio = avg_fps / max(1.0, native_fps)

    # Compute Statistics
    report = {
        "hardware": hw_info,
        "video": {
            "path": video_path,
            "resolution": f"{width}x{height}",
            "total_frames": total_video_frames,
            "processed_frames": processed_frames,
            "native_fps": round(native_fps, 2),
            "duration_sec": round(video_duration_sec, 2),
        },
        "performance": {
            "total_processing_time_sec": round(total_time_sec, 3),
            "average_fps": round(avg_fps, 2),
            "speedup_vs_realtime": f"{speedup_ratio:.2f}x ({'Faster than realtime' if speedup_ratio >= 1.0 else 'Slower than realtime'})",
            "avg_detection_latency_ms": round(float(np.mean(detect_times)), 2) if detect_times else 0.0,
            "min_detection_latency_ms": round(float(np.min(detect_times)), 2) if detect_times else 0.0,
            "max_detection_latency_ms": round(float(np.max(detect_times)), 2) if detect_times else 0.0,
            "avg_tracking_latency_ms": round(float(np.mean(track_times)), 2) if track_times else 0.0,
            "avg_e2e_frame_latency_ms": round(float(np.mean(e2e_times)), 2) if e2e_times else 0.0,
        },
        "memory_utilization": {
            "peak_client_process_ram_mb": round(peak_ram_mb, 2),
            "ram_used_percentage_of_4gb": round((peak_ram_mb / 4096.0) * 100, 2),
            "free_ram_headroom_gb": round(hw_info["total_ram_gb"] - (peak_ram_mb / 1024.0), 2)
        },
        "network_and_accuracy": {
            "total_cloud_dispatches": total_crop_dispatches,
            "avg_cloud_api_latency_ms": round(float(np.mean(api_latencies)), 2) if api_latencies else 0.0,
            "unique_employees_recognized": list(matched_identities),
            "total_recognized_count": len(matched_identities),
            "unknown_persons_flagged": unknown_incidents
        }
    }

    # Print Formatted Benchmark Table
    print("\n" + "=" * 80)
    print("🏆 FINAL BENCHMARK SUMMARY & PERFORMANCE PROFILE")
    print("=" * 80)
    print(f"💻 Machine CPU:               {hw_info['cpu_model']}")
    print(f"🧠 Client Process RAM Used:   {report['memory_utilization']['peak_client_process_ram_mb']} MB (Only {report['memory_utilization']['ram_used_percentage_of_4gb']}% of 4GB RAM)")
    print(f"🛡️  Free RAM Headroom:         {report['memory_utilization']['free_ram_headroom_gb']} GB")
    print(f"🎬 Video Duration:            {report['video']['duration_sec']} seconds ({report['video']['total_frames']} frames)")
    print(f"⏱️  Total Processing Time:     {report['performance']['total_processing_time_sec']} seconds")
    print(f"⚡ Average Processing Speed:  {report['performance']['average_fps']} FPS ({report['performance']['speedup_vs_realtime']})")
    print(f"🔬 Edge Detect Latency:       {report['performance']['avg_detection_latency_ms']} ms/frame")
    print(f"🔄 Edge Tracking Latency:     {report['performance']['avg_tracking_latency_ms']} ms/frame")
    print(f"🌐 Remote Recognition API:    {report['network_and_accuracy']['avg_cloud_api_latency_ms']} ms avg ({report['network_and_accuracy']['total_cloud_dispatches']} crops evaluated)")
    print(f"✅ Recognized Employees:      {len(matched_identities)} ({', '.join(matched_identities) if matched_identities else 'None'})")
    print(f"⚠️  Unknown Persons:           {unknown_incidents}")
    print("=" * 80)

    return report


def main():
    parser = argparse.ArgumentParser(description="AI Monk Edge Client Performance Benchmark")
    parser.add_argument("--video", type=str, default="phone_video.mp4", help="Path to video file")
    parser.add_argument("--server", type=str, default="https://49.206.228.75:9001", help="Remote GPU server URL")
    parser.add_argument("--model", type=str, default="models/ultra_light/version-RFB-320.onnx", help="Path to ONNX model")
    parser.add_argument("--frames", type=int, default=0, help="Max frames to profile (0 for entire video)")
    args = parser.parse_args()

    run_benchmark(args.video, args.server, args.model, args.frames)


if __name__ == "__main__":
    main()
