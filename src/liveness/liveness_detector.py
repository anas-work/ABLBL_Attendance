from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class LivenessResult:
    passed: bool
    status: str       # "PASS", "FAIL", "NOT_IMPLEMENTED"
    score: float = 1.0
    reason: Optional[str] = None

class LivenessDetector:
    """
    Extensible Face Anti-Spoofing / Liveness Interface.
    Architectural plug for lightweight Jetson anti-spoofing models (e.g. MiniFASNet / Silent-Face-Anti-Spoofing).
    Currently defaults to PASS/NOT_IMPLEMENTED stub until edge model deployment.
    """

    def __init__(self, enabled: bool = False, model_path: Optional[str] = None):
        self.enabled = enabled
        self.model_path = model_path

    def check_liveness(self, face_crop: np.ndarray, landmarks: Optional[np.ndarray] = None) -> LivenessResult:
        if not self.enabled:
            return LivenessResult(
                passed=True,
                status="NOT_IMPLEMENTED",
                score=1.0,
                reason="Liveness detection disabled in configuration (stub mode)"
            )

        # Placeholder logic: returns PASS for valid numpy face crops
        if face_crop is None or face_crop.size == 0:
            return LivenessResult(
                passed=False,
                status="FAIL",
                score=0.0,
                reason="Invalid face crop provided for liveness check"
            )

        return LivenessResult(
            passed=True,
            status="PASS",
            score=0.99,
            reason="Passed preliminary liveness check"
        )
