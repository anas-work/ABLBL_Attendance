import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

@dataclass
class TrackedFace:
    track_id: int
    bbox: np.ndarray             # [x1, y1, x2, y2]
    score: float
    kps: Optional[np.ndarray]    # 5x2 landmarks
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    assigned_identity: Optional[str] = None
    similarity_score: float = 0.0
    confirmed: bool = False
    recognition_stale: bool = True
    last_recognition_frame: int = 0

class IoUTracker:
    """
    Lightweight, high-performance Face Tracker (ByteTrack / IoU + Kalman approach).
    Assigns persistent track_id across consecutive frames to prevent per-frame re-recognition overhead.
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30, min_hits: int = 2):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.next_id = 1
        self.tracks: List[TrackedFace] = []
        self.frame_count = 0

    def update(self, detections: List[Any]) -> List[TrackedFace]:
        """
        Updates tracks with new frame detections.
        detections: list of objects having .bbox, .score, .kps
        """
        self.frame_count += 1

        # Increment age for existing tracks
        for t in self.tracks:
            t.age += 1
            t.time_since_update += 1

        if len(detections) == 0:
            # Remove stale tracks
            self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
            return [t for t in self.tracks if t.time_since_update == 0]

        det_boxes = np.array([d.bbox for d in detections]) if len(detections) > 0 else np.empty((0, 4))
        
        if len(self.tracks) == 0:
            # Initialize tracks for all detections
            matched_tracks = []
            for d in detections:
                new_t = TrackedFace(
                    track_id=self.next_id,
                    bbox=d.bbox,
                    score=d.score,
                    kps=d.kps,
                    hits=1,
                    age=1,
                    time_since_update=0
                )
                self.next_id += 1
                self.tracks.append(new_t)
                matched_tracks.append(new_t)
            return matched_tracks

        track_boxes = np.array([t.bbox for t in self.tracks])
        iou_matrix = self._compute_iou(track_boxes, det_boxes)

        matched_track_indices = []
        matched_det_indices = []

        if iou_matrix.size > 0:
            # Greedy max IoU matching
            while True:
                max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                max_iou = iou_matrix[max_idx]
                
                if max_iou < self.iou_threshold:
                    break

                t_idx, d_idx = max_idx
                if t_idx in matched_track_indices or d_idx in matched_det_indices:
                    iou_matrix[t_idx, d_idx] = -1.0
                    continue

                matched_track_indices.append(t_idx)
                matched_det_indices.append(d_idx)
                iou_matrix[t_idx, :] = -1.0
                iou_matrix[:, d_idx] = -1.0

        # Update matched tracks
        updated_tracks = []
        for t_idx, d_idx in zip(matched_track_indices, matched_det_indices):
            track = self.tracks[t_idx]
            det = detections[d_idx]

            # Smooth bbox transition (exponential moving average)
            track.bbox = 0.7 * det.bbox + 0.3 * track.bbox
            track.score = det.score
            track.kps = det.kps
            track.hits += 1
            track.time_since_update = 0
            
            # Mark for re-recognition periodically (every 30 frames) or if unassigned
            if self.frame_count - track.last_recognition_frame > 30 or track.assigned_identity is None:
                track.recognition_stale = True

            updated_tracks.append(track)

        # Create new tracks for unmatched detections
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_det_indices:
                new_t = TrackedFace(
                    track_id=self.next_id,
                    bbox=det.bbox,
                    score=det.score,
                    kps=det.kps,
                    hits=1,
                    age=1,
                    time_since_update=0,
                    recognition_stale=True
                )
                self.next_id += 1
                self.tracks.append(new_t)
                updated_tracks.append(new_t)

        # Prune dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        return [t for t in self.tracks if t.time_since_update == 0]

    @staticmethod
    def _compute_iou(boxesA: np.ndarray, boxesB: np.ndarray) -> np.ndarray:
        if len(boxesA) == 0 or len(boxesB) == 0:
            return np.empty((0, 0))

        xA = np.maximum(boxesA[:, 0][:, None], boxesB[:, 0][None, :])
        yA = np.maximum(boxesA[:, 1][:, None], boxesB[:, 1][None, :])
        xB = np.minimum(boxesA[:, 2][:, None], boxesB[:, 2][None, :])
        yB = np.minimum(boxesA[:, 3][:, None], boxesB[:, 3][None, :])

        interArea = np.maximum(0, xB - xA + 1) * np.maximum(0, yB - yA + 1)
        boxAArea = (boxesA[:, 2] - boxesA[:, 0] + 1) * (boxesA[:, 3] - boxesA[:, 1] + 1)
        boxBArea = (boxesB[:, 2] - boxesB[:, 0] + 1) * (boxesB[:, 3] - boxesB[:, 1] + 1)

        iou = interArea / (boxAArea[:, None] + boxBArea[None, :] - interArea)
        return iou
