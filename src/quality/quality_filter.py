import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

@dataclass
class QualityResult:
    passed: bool
    score: float
    reason: Optional[str] = None
    sharpness: float = 0.0
    brightness: float = 0.0
    face_size: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0

class FaceQualityFilter:
    """
    Evaluates detected face quality against configurable thresholds:
    - Minimum face size (width/height in pixels)
    - Detection confidence score
    - Image sharpness (Laplacian variance)
    - Illumination / Brightness range
    - Head pose (Yaw, Pitch, Roll angles estimated from 5 landmarks)
    """

    def __init__(self, config: Dict[str, Any]):
        q_cfg = config.get("quality", {})
        self.min_face_size = q_cfg.get("min_face_size", 120)
        self.min_det_conf = q_cfg.get("min_detection_confidence", 0.30)
        self.min_sharpness = q_cfg.get("min_sharpness", 25.0)
        self.min_brightness = q_cfg.get("min_brightness", 5.0)
        self.max_brightness = q_cfg.get("max_brightness", 250.0)
        self.max_yaw = q_cfg.get("max_yaw", 55.0)
        self.max_pitch = q_cfg.get("max_pitch", 45.0)
        self.max_roll = q_cfg.get("max_roll", 45.0)

    def evaluate(
        self,
        image: np.ndarray,
        bbox: np.ndarray,
        det_score: float,
        landmarks: Optional[np.ndarray] = None
    ) -> QualityResult:
        x1, y1, x2, y2 = bbox.astype(int)
        h_img, w_img = image.shape[:2]

        # Clamp crop coordinates
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)
        
        face_w = x2 - x1
        face_h = y2 - y1
        min_dim = min(face_w, face_h)

        if min_dim < self.min_face_size:
            return QualityResult(
                passed=False,
                score=0.0,
                reason=f"Face size too small ({min_dim}px < {self.min_face_size}px)",
                face_size=float(min_dim)
            )

        if det_score < self.min_det_conf:
            return QualityResult(
                passed=False,
                score=0.0,
                reason=f"Low detection confidence ({det_score:.2f} < {self.min_det_conf:.2f})",
                face_size=float(min_dim)
            )

        face_crop = image[y1:y2, x1:x2]
        if face_crop.size == 0:
            return QualityResult(passed=False, score=0.0, reason="Empty face crop")

        gray_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if face_crop.ndim == 3 else face_crop

        # Sharpness via Laplacian variance
        sharpness = float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())
        if sharpness < self.min_sharpness:
            return QualityResult(
                passed=False,
                score=0.0,
                reason=f"Image too blurry (sharpness {sharpness:.1f} < {self.min_sharpness:.1f})",
                sharpness=sharpness,
                face_size=float(min_dim)
            )

        # Brightness / Illumination check
        brightness = float(np.mean(gray_crop))
        if brightness < self.min_brightness or brightness > self.max_brightness:
            return QualityResult(
                passed=False,
                score=0.0,
                reason=f"Poor illumination (brightness {brightness:.1f} outside [{self.min_brightness}, {self.max_brightness}])",
                sharpness=sharpness,
                brightness=brightness,
                face_size=float(min_dim)
            )

        # Pose Estimation from 5 Keypoint Landmarks
        yaw, pitch, roll = 0.0, 0.0, 0.0
        if landmarks is not None and len(landmarks) == 5:
            yaw, pitch, roll = self._estimate_pose_5kps(landmarks)
            if abs(yaw) > self.max_yaw:
                return QualityResult(
                    passed=False,
                    score=0.0,
                    reason=f"Extreme yaw angle ({yaw:.1f}° > {self.max_yaw}°)",
                    sharpness=sharpness,
                    brightness=brightness,
                    face_size=float(min_dim),
                    yaw=yaw, pitch=pitch, roll=roll
                )
            if abs(pitch) > self.max_pitch:
                return QualityResult(
                    passed=False,
                    score=0.0,
                    reason=f"Extreme pitch angle ({pitch:.1f}° > {self.max_pitch}°)",
                    sharpness=sharpness,
                    brightness=brightness,
                    face_size=float(min_dim),
                    yaw=yaw, pitch=pitch, roll=roll
                )

        # Overall Quality Score (0 to 1)
        sharp_factor = min(1.0, sharpness / (self.min_sharpness * 3.0))
        size_factor = min(1.0, min_dim / 150.0)
        overall_score = float(0.4 * det_score + 0.3 * sharp_factor + 0.3 * size_factor)

        return QualityResult(
            passed=True,
            score=overall_score,
            sharpness=sharpness,
            brightness=brightness,
            face_size=float(min_dim),
            yaw=yaw, pitch=pitch, roll=roll
        )

    @staticmethod
    def _estimate_pose_5kps(kps: np.ndarray) -> Tuple[float, float, float]:
        """
        Rough pose estimation (yaw, pitch, roll in degrees) from 5 landmarks.
        kps: [left_eye, right_eye, nose, left_mouth, right_mouth]
        """
        le, re, nose, lm, rm = kps[0], kps[1], kps[2], kps[3], kps[4]

        # Roll: Eye tilt angle
        dx = re[0] - le[0]
        dy = re[1] - le[1]
        roll = np.degrees(np.arctan2(dy, dx))

        # Yaw: Eye-to-nose horizontal ratio asymmetry
        dist_l_nose = np.linalg.norm(nose - le)
        dist_r_nose = np.linalg.norm(nose - re)
        if (dist_l_nose + dist_r_nose) > 0:
            ratio = (dist_r_nose - dist_l_nose) / (dist_l_nose + dist_r_nose)
            yaw = float(ratio * 90.0)
        else:
            yaw = 0.0

        # Pitch: Eye center to nose vertical distance vs nose to mouth center
        eye_center = (le + re) / 2.0
        mouth_center = (lm + rm) / 2.0
        d_eye_nose = np.linalg.norm(nose - eye_center)
        d_nose_mouth = np.linalg.norm(mouth_center - nose)
        if d_nose_mouth > 0:
            pitch_ratio = (d_eye_nose / d_nose_mouth) - 1.0
            pitch = float(pitch_ratio * 45.0)
        else:
            pitch = 0.0

        return float(yaw), float(pitch), float(roll)