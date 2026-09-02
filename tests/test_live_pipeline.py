import sys
import yaml
import time
import cv2
from src.video.file_source import FileVideoSource
from src.pipeline import RecognitionPipeline

def test_pipeline():
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Disable loop for testing single pass over 100 frames
    source = FileVideoSource("Employees_Video/reference_video.mp4", loop=False)
    pipeline = RecognitionPipeline(config=config, video_source=source)

    print("\n--- RUNNING RECOGNITION PIPELINE ON SIMULATED LIVE STREAM ---")
    frame_count = 0
    total_latency = 0.0

    while True:
        ret, frame = source.read()
        if not ret or frame is None or frame_count >= 60:
            break

        res = pipeline.process_frame(frame)
        frame_count += 1
        total_latency += res.latency_breakdown["end_to_end_ms"]

        print(
            f"Frame #{res.frame_index:03d} | "
            f"FPS: {res.fps:.1f} | "
            f"Detect: {res.latency_breakdown['detection_ms']:.1f}ms | "
            f"Rec: {res.latency_breakdown['recognition_ms']:.1f}ms | "
            f"Total: {res.latency_breakdown['end_to_end_ms']:.1f}ms | "
            f"Active Tracks: {len(res.active_tracks)}"
        )

        for dec in res.decisions:
            print(f"   -> Track #{dec.track_id}: Decision={dec.decision}, Employee={dec.name} ({dec.employee_id}), Conf={dec.confidence*100:.1f}%")

    source.release()
    avg_latency = total_latency / frame_count if frame_count > 0 else 0.0
    print(f"\nCompleted {frame_count} frames. Average End-to-End Latency: {avg_latency:.2f} ms")

if __name__ == "__main__":
    test_pipeline()
