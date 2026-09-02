import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

@dataclass
class TemporalDecision:
    track_id: int
    decision: str              # "VERIFIED", "UNKNOWN", "LOW_QUALITY", "UNCERTAIN"
    employee_id: Optional[str] = None
    name: Optional[str] = None
    confidence: float = 0.0
    agreement_ratio: float = 0.0
    confirmed: bool = False

class TemporalConfirmationEngine:
    """
    Temporal confirmation state machine.
    Prevents single-frame identity flicker by maintaining a sliding window of candidate matches
    per track ID and requiring configurable multi-frame consensus before issuing a VERIFIED decision.
    """

    def __init__(self, config: Dict[str, Any]):
        t_cfg = config.get("temporal", {})
        r_cfg = config.get("recognition", {})

        self.confirmation_count = t_cfg.get("confirmation_count", 3)
        self.window_size = t_cfg.get("temporal_window_frames", 10)
        self.min_avg_sim = t_cfg.get("min_average_similarity", 0.48)
        self.min_agreement_ratio = t_cfg.get("min_agreement_ratio", 0.6)
        self.match_threshold = r_cfg.get("match_threshold", 0.45)
        self.unknown_threshold = r_cfg.get("unknown_threshold", 0.35)

        # Map: track_id -> deque of (candidate_emp_id, candidate_name, sim_score)
        self.history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.window_size))
        # Map: track_id -> finalized TemporalDecision
        self.confirmed_decisions: Dict[int, TemporalDecision] = {}

    def update(
        self,
        track_id: int,
        raw_match: Optional[Tuple[float, Dict[str, str]]],
        quality_passed: bool = True
    ) -> TemporalDecision:
        """
        Updates track history with frame recognition match and returns TemporalDecision.
        """
        # If face is low quality, record low quality marker
        if not quality_passed or raw_match is None:
            self.history[track_id].append(("LOW_QUALITY", "Low Quality", 0.0))
            return self._evaluate_history(track_id)

        sim_score, meta = raw_match

        if sim_score < self.unknown_threshold:
            candidate_id = "UNKNOWN"
            candidate_name = "Unknown Person"
        elif sim_score < self.match_threshold:
            candidate_id = "UNCERTAIN"
            candidate_name = "Uncertain Match"
        else:
            candidate_id = meta.get("employee_id", "UNKNOWN")
            candidate_name = meta.get("name", "Unknown Person")

        self.history[track_id].append((candidate_id, candidate_name, sim_score))
        return self._evaluate_history(track_id)

    def _evaluate_history(self, track_id: int) -> TemporalDecision:
        window = self.history[track_id]
        if not window:
            return TemporalDecision(track_id=track_id, decision="UNCERTAIN")

        # Count candidate frequency in window
        counts: Dict[str, int] = defaultdict(int)
        sim_scores: Dict[str, List[float]] = defaultdict(list)
        names: Dict[str, str] = {}

        for emp_id, name, score in window:
            counts[emp_id] += 1
            sim_scores[emp_id].append(score)
            names[emp_id] = name

        # Find top candidate
        top_emp_id = max(counts, key=counts.get)
        top_count = counts[top_emp_id]
        agreement_ratio = top_count / len(window)
        avg_sim = float(np.mean(sim_scores[top_emp_id])) if sim_scores[top_emp_id] else 0.0

        if top_emp_id == "LOW_QUALITY":
            return TemporalDecision(
                track_id=track_id,
                decision="LOW_QUALITY",
                confidence=0.0,
                agreement_ratio=agreement_ratio,
                confirmed=False
            )

        if top_emp_id == "UNKNOWN" or avg_sim < self.unknown_threshold:
            return TemporalDecision(
                track_id=track_id,
                decision="UNKNOWN",
                employee_id="UNKNOWN",
                name="Unknown Person",
                confidence=avg_sim,
                agreement_ratio=agreement_ratio,
                confirmed=(top_count >= self.confirmation_count)
            )

        # Check threshold for VERIFIED employee
        if (
            top_count >= self.confirmation_count and
            agreement_ratio >= self.min_agreement_ratio and
            avg_sim >= self.min_avg_sim
        ):
            decision = TemporalDecision(
                track_id=track_id,
                decision="VERIFIED",
                employee_id=top_emp_id,
                name=names.get(top_emp_id, top_emp_id),
                confidence=avg_sim,
                agreement_ratio=agreement_ratio,
                confirmed=True
            )
            self.confirmed_decisions[track_id] = decision
            return decision

        return TemporalDecision(
            track_id=track_id,
            decision="UNCERTAIN",
            employee_id=top_emp_id,
            name=names.get(top_emp_id, top_emp_id),
            confidence=avg_sim,
            agreement_ratio=agreement_ratio,
            confirmed=False
        )

    def cleanup_track(self, track_id: int) -> None:
        """Cleans up history for expired tracks."""
        if track_id in self.history:
            del self.history[track_id]
        if track_id in self.confirmed_decisions:
            del self.confirmed_decisions[track_id]
