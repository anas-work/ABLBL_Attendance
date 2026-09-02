import re
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class EmployeeMeta:
    filename: str
    name: str
    employee_id: str
    valid: bool
    error_reason: Optional[str] = None

class EmployeeFilenameParser:
    """
    Parses employee photograph filenames into employee name and ID.
    Handles irregular spaces, dots in extensions, e.g.:
    "Rahul Sharma EMP001.jpg" -> Name: "Rahul Sharma", ID: "EMP001"
    "Hanumantha Raju ABL886..jpg" -> Name: "Hanumantha Raju", ID: "ABL886"
    """
    
    # Matches <Name> <Space/Tabs> <EmployeeID>.<ext>
    # ID is alphanumeric e.g. ABL188, RAD0712, RADT004, RDX006
    PATTERN = re.compile(r'^(.*?)\s+([A-Za-z0-9]+)\.*(?:jpg|jpeg|png)$', re.IGNORECASE)

    @classmethod
    def parse(cls, filepath_or_name: str) -> EmployeeMeta:
        filename = os.path.basename(filepath_or_name).strip()
        
        # Clean extra dots before extension if any
        clean_name = re.sub(r'\.+(?:jpg|jpeg|png)$', '.jpg', filename, flags=re.IGNORECASE)
        
        match = cls.PATTERN.match(clean_name)
        if not match:
            # Try fallback split by last whitespace
            base_name, _ = os.path.splitext(filename)
            base_name = base_name.rstrip('.')
            parts = base_name.rsplit(maxsplit=1)
            if len(parts) == 2 and re.match(r'^[A-Za-z0-9]+$', parts[1]):
                name = parts[0].strip()
                emp_id = parts[1].strip()
                return EmployeeMeta(
                    filename=filename,
                    name=name,
                    employee_id=emp_id,
                    valid=True
                )
            
            return EmployeeMeta(
                filename=filename,
                name="",
                employee_id="",
                valid=False,
                error_reason="Could not parse Name and Employee ID pattern from filename"
            )
            
        name = match.group(1).strip()
        emp_id = match.group(2).strip()
        
        if not name or not emp_id:
            return EmployeeMeta(
                filename=filename,
                name=name,
                employee_id=emp_id,
                valid=False,
                error_reason="Empty name or employee_id after parsing"
            )
            
        return EmployeeMeta(
            filename=filename,
            name=name,
            employee_id=emp_id,
            valid=True
        )
