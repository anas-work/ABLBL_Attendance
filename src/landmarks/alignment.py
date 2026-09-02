import cv2
import numpy as np
from typing import Tuple, Optional

# Standard 112x112 reference landmarks for AdaFace / KPRPE / ArcFace alignment
REFERENCE_FACIAL_LANDMARKS_112x112 = np.array([
    [38.2946, 51.6963],  # Left Eye
    [73.5318, 51.5014],  # Right Eye
    [56.0252, 71.7366],  # Nose Tip
    [41.5493, 92.3655],  # Left Mouth Corner
    [70.7299, 92.2041]   # Right Mouth Corner
], dtype=np.float32)

class FaceAligner:
    """
    5-Point Facial Landmark Similarity Transformation Engine (Umeyama Algorithm).
    Aligns and normalizes face crops to 112x112 for KPRPE + AdaFace feature extraction.
    """

    @staticmethod
    def align_face_112(image: np.ndarray, landmarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aligns a face image to 112x112 using 5 facial keypoints.
        
        Args:
            image: Input image (BGR or RGB, uint8)
            landmarks: 5x2 numpy array of (x, y) coordinates for:
                       [left_eye, right_eye, nose, left_mouth, right_mouth]
                       
        Returns:
            aligned_face: 112x112 aligned RGB/BGR image.
            transform_matrix: 2x3 affine transformation matrix.
        """
        if landmarks is None or len(landmarks) != 5:
            raise ValueError("Exactly 5 facial keypoints are required for alignment.")

        src_pts = np.array(landmarks, dtype=np.float32)
        dst_pts = REFERENCE_FACIAL_LANDMARKS_112x112

        # Compute optimal similarity transformation (rigid: rotation + scale + translation)
        tfm, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)
        
        if tfm is None:
            # Fallback to standard affine if partial 2D fails
            tfm = cv2.getAffineTransform(src_pts[:3], dst_pts[:3])

        aligned_face = cv2.warpAffine(image, tfm, (112, 112), flags=cv2.INTER_LINEAR)
        return aligned_face, tfm
