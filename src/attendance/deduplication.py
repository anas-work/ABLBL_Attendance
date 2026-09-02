import time
from dataclasses import dataclass
from typing import Dict, Optional, Any

@dataclass
class AttendanceEvent:
    employee_id: str
    name: str
    camera_id: str
    timestamp: float
    formatted_time: str
    similarity: float

class AttendanceDeduplicator:
    """
    Attendance Event Deduplicator & Cooldown Controller.
    Prevents flooding the database with duplicate records for the same employee track.
    """

    def __init__(self, cooldown_seconds: float = 30.0):
        self.cooldown_seconds = cooldown_seconds
        # Map: employee_id -> last_attendance_timestamp
        self.last_recorded: Dict[str, float] = {}

    def should_record(self, employee_id: str, current_time: Optional[float] = None) -> bool:
        """
        Returns True if an attendance event should be recorded for employee_id.
        Enforces a minimum cooldown (default 10s) between events for the same employee.
        """
        if not employee_id or employee_id in ["UNKNOWN", "UNCERTAIN", "LOW_QUALITY"]:
            return False

        now = current_time or time.time()
        last_time = self.last_recorded.get(employee_id, 0.0)

        if (now - last_time) >= self.cooldown_seconds:
            self.last_recorded[employee_id] = now
            return True

        return False

    def get_last_recorded_time(self, employee_id: str) -> float:
        """Returns the last timestamp when an event was recorded for employee_id."""
        return self.last_recorded.get(employee_id, 0.0)

    def reset_cooldown(self, employee_id: str) -> None:
        """Clears cooldown entry for employee_id."""
        if employee_id in self.last_recorded:
            del self.last_recorded[employee_id]

    def clear_all(self) -> None:
        """Clears all cooldown records across all employees upon mode switch."""
        self.last_recorded.clear()

