import os
import json
import cv2
import time
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

from src.enrollment.parser import EmployeeFilenameParser
from src.detection.scrfd_detector import SCRFDDetector
from src.landmarks.alignment import FaceAligner
from src.recognition.kprpe_adaface import KPRPEAdaFaceRecognizer
from src.search.faiss_index import FAISSVectorIndex

@dataclass
class EnrollmentRecord:
    filename: str
    employee_id: str
    name: str
    status: str       # "SUCCESS" or "FAILED"
    reason: Optional[str] = None
    face_score: Optional[float] = None
    embedding_norm: Optional[float] = None

class EnrollmentService:
    """
    Scans photo directory, validates employee metadata, detects faces,
    aligns keypoints, computes KPRPE+AdaFace embeddings, and indexes into FAISS gallery.
    Generates structured enrollment summary reports.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        detector: Optional[SCRFDDetector] = None,
        recognizer: Optional[KPRPEAdaFaceRecognizer] = None,
        index: Optional[FAISSVectorIndex] = None
    ):
        self.config = config
        
        models_cfg = config.get("detection", {})
        rec_cfg = config.get("recognition", {})
        storage_cfg = config.get("storage", {})

        self.photos_dir = storage_cfg.get("photos_dir", "Employees_Photo")
        self.reports_dir = storage_cfg.get("reports_dir", "data/enrollment_reports")
        os.makedirs(self.reports_dir, exist_ok=True)

        self.detector = detector or SCRFDDetector(
            model_path=models_cfg.get("model_path", "models/scrfd_2.5g_kps.onnx"),
            conf_threshold=models_cfg.get("confidence_threshold", 0.5)
        )

        self.recognizer = recognizer or KPRPEAdaFaceRecognizer(
            model_path=rec_cfg.get("model_path", "models/kprpe_adaface.onnx")
        )

        self.index = index or FAISSVectorIndex(
            dimension=rec_cfg.get("embedding_size", 512),
            index_dir=storage_cfg.get("gallery_dir", "data/embeddings")
        )

    def run_enrollment(self) -> Dict[str, Any]:
        """
        Executes complete employee enrollment over photos_dir.
        """
        start_time = time.time()
        if not os.path.exists(self.photos_dir):
            raise FileNotFoundError(f"Photos directory not found: {self.photos_dir}")

        image_files = [
            f for f in os.listdir(self.photos_dir)
            if not f.startswith('.') and f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        records: List[EnrollmentRecord] = []
        embeddings_to_add = []
        metadata_to_add = []

        print(f"Starting enrollment for {len(image_files)} employee photo(s)...")

        for fname in sorted(image_files):
            filepath = os.path.join(self.photos_dir, fname)
            meta = EmployeeFilenameParser.parse(fname)

            if not meta.valid:
                records.append(EnrollmentRecord(
                    filename=fname,
                    employee_id="",
                    name="",
                    status="FAILED",
                    reason=meta.error_reason
                ))
                continue

            # Load image
            img = cv2.imread(filepath)
            if img is None:
                records.append(EnrollmentRecord(
                    filename=fname,
                    employee_id=meta.employee_id,
                    name=meta.name,
                    status="FAILED",
                    reason="Failed to read or decode image file"
                ))
                continue

            # Detect faces
            try:
                detections = self.detector.detect(img)
            except Exception as e:
                records.append(EnrollmentRecord(
                    filename=fname,
                    employee_id=meta.employee_id,
                    name=meta.name,
                    status="FAILED",
                    reason=f"Detection error: {str(e)}"
                ))
                continue

            if len(detections) == 0:
                records.append(EnrollmentRecord(
                    filename=fname,
                    employee_id=meta.employee_id,
                    name=meta.name,
                    status="FAILED",
                    reason="No face detected in photo"
                ))
                continue

            if len(detections) > 1:
                # Select face with highest confidence score
                detections.sort(key=lambda d: d.score, reverse=True)

            best_det = detections[0]

            if best_det.kps is None or len(best_det.kps) != 5:
                records.append(EnrollmentRecord(
                    filename=fname,
                    employee_id=meta.employee_id,
                    name=meta.name,
                    status="FAILED",
                    reason="Facial keypoint landmarks missing or incomplete"
                ))
                continue

            # Align face to 112x112 using 5 keypoint landmarks
            try:
                aligned_crop, _ = FaceAligner.align_face_112(img, best_det.kps)
                embedding = self.recognizer.extract_embedding(aligned_crop)
            except Exception as e:
                records.append(EnrollmentRecord(
                    filename=fname,
                    employee_id=meta.employee_id,
                    name=meta.name,
                    status="FAILED",
                    reason=f"Embedding extraction error: {str(e)}"
                ))
                continue

            emb_norm = float(np.linalg.norm(embedding))

            embeddings_to_add.append(embedding)
            metadata_to_add.append({
                "employee_id": meta.employee_id,
                "name": meta.name,
                "filename": fname,
                "image_path": filepath
            })

            records.append(EnrollmentRecord(
                filename=fname,
                employee_id=meta.employee_id,
                name=meta.name,
                status="SUCCESS",
                face_score=float(best_det.score),
                embedding_norm=emb_norm
            ))

        # Add to FAISS index and save
        if len(embeddings_to_add) > 0:
            embeddings_matrix = np.vstack(embeddings_to_add)
            self.index.add_embeddings(embeddings_matrix, metadata_to_add)
            self.index.save()

        elapsed = time.time() - start_time
        success_count = sum(1 for r in records if r.status == "SUCCESS")
        failed_count = sum(1 for r in records if r.status == "FAILED")

        summary = {
            "total_processed": len(image_files),
            "enrolled_success": success_count,
            "enrolled_failed": failed_count,
            "total_vectors_in_gallery": self.index.total_vectors,
            "elapsed_seconds": round(elapsed, 2),
            "records": [asdict(r) for r in records]
        }

        # Write enrollment report JSON
        report_file = os.path.join(self.reports_dir, "enrollment_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

        print(f"Enrollment Complete: {success_count}/{len(image_files)} photos enrolled successfully in {elapsed:.2f}s.")
        print(f"Report saved to {report_file}")

        return summary

    def enroll_single(
        self,
        name: str,
        employee_id: str,
        image_input: Any,
        department: str = "General"
    ) -> Dict[str, Any]:
        """
        Enrolls a single employee photo into Employees_Photo/ and updates FAISS index in real time.
        `image_input` can be raw bytes, a filepath string, or a numpy BGR ndarray.
        """
        clean_name = name.strip()
        clean_emp_id = employee_id.strip()

        if not clean_name:
            raise ValueError("Employee Name cannot be empty.")
        if not clean_emp_id:
            raise ValueError("Employee ID cannot be empty.")

        # Decode image
        if isinstance(image_input, (bytes, bytearray)):
            jpg_arr = np.frombuffer(image_input, dtype=np.uint8)
            img = cv2.imdecode(jpg_arr, cv2.IMREAD_COLOR)
        elif isinstance(image_input, str):
            img = cv2.imread(image_input)
        elif isinstance(image_input, np.ndarray):
            img = image_input
        else:
            raise ValueError("Invalid image input format.")

        if img is None:
            raise ValueError("Failed to decode or read the uploaded photo.")

        # Detect face
        detections = self.detector.detect(img)
        if len(detections) == 0:
            raise ValueError("No face detected in the photo. Please upload a clear frontal photo.")

        detections.sort(key=lambda d: d.score, reverse=True)
        best_det = detections[0]

        if best_det.kps is None or len(best_det.kps) != 5:
            raise ValueError("Facial landmark keypoints could not be extracted.")

        # Align face to 112x112
        aligned_crop, _ = FaceAligner.align_face_112(img, best_det.kps)
        embedding = self.recognizer.extract_embedding(aligned_crop)

        # Standard filename convention matching existing gallery
        filename = f"{clean_name} {clean_emp_id}.jpg"
        save_path = os.path.join(self.photos_dir, filename)
        os.makedirs(self.photos_dir, exist_ok=True)
        cv2.imwrite(save_path, img)

        # Update FAISS gallery
        metadata = {
            "employee_id": clean_emp_id,
            "name": clean_name,
            "filename": filename,
            "image_path": save_path
        }
        self.index.add_embeddings(np.array([embedding]), [metadata])
        self.index.save()

        return {
            "status": "SUCCESS",
            "employee_id": clean_emp_id,
            "name": clean_name,
            "department": department,
            "filename": filename,
            "image_path": save_path,
            "photo_url": f"/photos/{filename}",
            "face_score": float(best_det.score),
            "total_vectors": self.index.total_vectors,
            "embedding": embedding,
            "bgr_image": img
        }
