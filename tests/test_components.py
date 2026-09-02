import pytest
import numpy as np
import os
import yaml

from src.enrollment.parser import EmployeeFilenameParser
from src.landmarks.alignment import FaceAligner
from src.quality.quality_filter import FaceQualityFilter
from src.search.faiss_index import FAISSVectorIndex
from src.temporal.temporal_confirmation import TemporalConfirmationEngine
from src.attendance.deduplication import AttendanceDeduplicator

def test_filename_parser_standard():
    meta = EmployeeFilenameParser.parse("Rahul Sharma EMP001.jpg")
    assert meta.valid is True
    assert meta.name == "Rahul Sharma"
    assert meta.employee_id == "EMP001"

def test_filename_parser_double_dot():
    meta = EmployeeFilenameParser.parse("Hanumantha Raju ABL886..jpg")
    assert meta.valid is True
    assert meta.name == "Hanumantha Raju"
    assert meta.employee_id == "ABL886"

def test_landmark_alignment_shape():
    dummy_img = np.zeros((400, 400, 3), dtype=np.uint8)
    dummy_kps = np.array([
        [150, 150], [250, 150], [200, 200], [170, 260], [230, 260]
    ], dtype=np.float32)

    crop, tfm = FaceAligner.align_face_112(dummy_img, dummy_kps)
    assert crop.shape == (112, 112, 3)
    assert tfm.shape == (2, 3)

def test_faiss_vector_search():
    index = FAISSVectorIndex(dimension=512, index_dir="data/test_embeddings")
    
    vec1 = np.random.randn(512).astype(np.float32)
    vec1 = vec1 / np.linalg.norm(vec1)
    
    vec2 = np.random.randn(512).astype(np.float32)
    vec2 = vec2 / np.linalg.norm(vec2)

    embeddings = np.vstack([vec1, vec2])
    metadata = [
        {"employee_id": "EMP001", "name": "Rahul Sharma"},
        {"employee_id": "EMP002", "name": "Amit Kumar"}
    ]

    index.add_embeddings(embeddings, metadata)
    assert index.total_vectors == 2

    # Query with exact vec1
    matches = index.search(vec1, top_k=1)
    assert len(matches) == 1
    sim, meta = matches[0]
    assert meta["employee_id"] == "EMP001"
    assert sim > 0.99

def test_temporal_confirmation_consensus():
    config = {
        "temporal": {
            "confirmation_count": 3,
            "temporal_window_frames": 5,
            "min_average_similarity": 0.50,
            "min_agreement_ratio": 0.6
        },
        "recognition": {
            "match_threshold": 0.45,
            "unknown_threshold": 0.35
        }
    }
    engine = TemporalConfirmationEngine(config)
    
    match_data = (0.85, {"employee_id": "EMP001", "name": "Rahul Sharma"})

    # Frame 1
    d1 = engine.update(track_id=1, raw_match=match_data, quality_passed=True)
    assert d1.decision == "UNCERTAIN"

    # Frame 2
    d2 = engine.update(track_id=1, raw_match=match_data, quality_passed=True)
    assert d2.decision == "UNCERTAIN"

    # Frame 3 (Hit threshold)
    d3 = engine.update(track_id=1, raw_match=match_data, quality_passed=True)
    assert d3.decision == "VERIFIED"
    assert d3.employee_id == "EMP001"

def test_attendance_deduplication_cooldown():
    dedup = AttendanceDeduplicator(cooldown_seconds=10.0)
    now = 1000.0

    # First event -> should record
    assert dedup.should_record("EMP001", current_time=now) is True

    # Immediate duplicate -> should ignore
    assert dedup.should_record("EMP001", current_time=now + 2.0) is False

    # After cooldown -> should record again
    assert dedup.should_record("EMP001", current_time=now + 12.0) is True
