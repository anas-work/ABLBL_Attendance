import os
import sys
import time
import glob
import yaml
import cv2
import numpy as np
import psutil
import torch
import onnxruntime as ort

from src.detection.scrfd_detector import SCRFDDetector
from src.recognition.kprpe_adaface import KPRPEAdaFaceRecognizer
from src.pipeline import RecognitionPipeline

def compute_percentiles(latencies_ms: list) -> dict:
    if not latencies_ms:
        return {"avg": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "fps": 0.0}
    arr = np.array(latencies_ms)
    return {
        "avg": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "fps": float(1000.0 / np.mean(arr)) if np.mean(arr) > 0 else 0.0
    }

def benchmark_ultralight(model_path: str = "models/ultra_light/version-RFB-320.onnx", iterations: int = 300):
    print(f"\n[1/5] Benchmarking Client-Side Ultra-Light RFB-320 ONNX Detector ({iterations} iterations)...")
    if not os.path.exists(model_path):
        print("  Skipping: Ultra-Light ONNX model not found.")
        return {}
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
    sess = ort.InferenceSession(model_path, providers=providers)
    input_name = sess.get_inputs()[0].name
    dummy_input = np.random.randn(1, 3, 240, 320).astype(np.float32)

    # Warmup
    for _ in range(20):
        sess.run(None, {input_name: dummy_input})

    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        sess.run(None, {input_name: dummy_input})
        latencies.append((time.perf_counter() - t0) * 1000.0)

    stats = compute_percentiles(latencies)
    print(f"  Ultra-Light Latency -> Mean: {stats['avg']:.2f}ms | p50: {stats['p50']:.2f}ms | p95: {stats['p95']:.2f}ms | p99: {stats['p99']:.2f}ms | FPS: {stats['fps']:.1f}")
    return stats

def benchmark_scrfd(detector: SCRFDDetector, iterations: int = 200):
    print(f"\n[2/5] Benchmarking SCRFD 5-Landmark Face Alignment ({iterations} iterations)...")
    dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Warmup
    for _ in range(15):
        detector.detect(dummy_img)

    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        detector.detect(dummy_img)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    stats = compute_percentiles(latencies)
    print(f"  SCRFD Latency -> Mean: {stats['avg']:.2f}ms | p50: {stats['p50']:.2f}ms | p95: {stats['p95']:.2f}ms | p99: {stats['p99']:.2f}ms | FPS: {stats['fps']:.1f}")
    return stats

def benchmark_adaface(recognizer: KPRPEAdaFaceRecognizer, iterations: int = 300):
    print(f"\n[3/5] Benchmarking AdaFace 512-d GPU Feature Extractor ({iterations} iterations)...")
    dummy_face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)

    # Warmup
    for _ in range(20):
        recognizer.extract_embedding(dummy_face)

    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        recognizer.extract_embedding(dummy_face)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    stats = compute_percentiles(latencies)
    print(f"  AdaFace Latency -> Mean: {stats['avg']:.2f}ms | p50: {stats['p50']:.2f}ms | p95: {stats['p95']:.2f}ms | p99: {stats['p99']:.2f}ms | FPS: {stats['fps']:.1f}")
    return stats

def benchmark_faiss(pipeline: RecognitionPipeline, iterations: int = 500):
    print(f"\n[4/5] Benchmarking FAISS Vector Search across {pipeline.gallery.total_vectors} enrolled employees ({iterations} iterations)...")
    dummy_emb = np.random.randn(512).astype(np.float32)
    dummy_emb /= np.linalg.norm(dummy_emb)

    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        pipeline.gallery.search(dummy_emb, top_k=1)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    stats = compute_percentiles(latencies)
    print(f"  FAISS Latency -> Mean: {stats['avg']:.4f}ms | p50: {stats['p50']:.4f}ms | p95: {stats['p95']:.4f}ms | p99: {stats['p99']:.4f}ms | FPS: {stats['fps']:.1f}")
    return stats

def evaluate_accuracy(pipeline: RecognitionPipeline):
    print(f"\n[5/5] Evaluating Top-1 Recognition Accuracy on Enrolled Workforce...")
    photos = glob.glob("Employees_Photo/*.jpg") + glob.glob("Employees_Photo/*.jpeg") + glob.glob("Employees_Photo/*.png")
    if not photos:
        print("  No photos found in Employees_Photo/")
        return

    correct = 0
    total = 0
    sims = []

    for p in photos:
        fname = os.path.basename(p)
        name_part = os.path.splitext(fname)[0]
        expected_id = name_part.split()[-1] if len(name_part.split()) > 1 else name_part
        img = cv2.imread(p)
        if img is None: continue

        success, enc = cv2.imencode('.jpg', img)
        if not success: continue

        res = pipeline.process_crop(enc.tobytes(), None)
        total += 1
        if res.get("matched"):
            sim = res.get("similarity", 0.0)
            sims.append(sim)
            matched_id = res.get("employee_id")
            if matched_id == expected_id or expected_id in res.get("name", ""):
                correct += 1

    acc = (correct / max(1, total)) * 100.0
    print(f"  Top-1 Match Accuracy: {acc:.2f}% ({correct}/{total} photos correctly matched)")
    if sims:
        print(f"  Cosine Similarity Score -> Mean: {np.mean(sims):.4f} | Min: {np.min(sims):.4f} | Max: {np.max(sims):.4f} | Std: {np.std(sims):.4f}")

def main():
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    pipeline = RecognitionPipeline(config=config)
    detector = SCRFDDetector(model_path=config["detection"]["model_path"])
    recognizer = KPRPEAdaFaceRecognizer(model_path=config["recognition"]["model_path"])

    print("================================================================================")
    print("      AI MONK ATTENDANCE SYSTEM — PRODUCTION BENCHMARK & ACCURACY SUITE         ")
    print("================================================================================")

    benchmark_ultralight()
    benchmark_scrfd(detector)
    benchmark_adaface(recognizer)
    benchmark_faiss(pipeline)
    evaluate_accuracy(pipeline)

    print("\n================================================================================")
    print("  Benchmark Suite Completed Successfully.")
    print("================================================================================")

if __name__ == "__main__":
    main()
