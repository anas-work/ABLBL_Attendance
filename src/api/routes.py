import asyncio
import cv2
import json
import time
from typing import Dict, Any, List
from fastapi import APIRouter, Response, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
import numpy as np

router = APIRouter()

# Global pipeline reference set in app.py startup
pipeline_instance = None

def get_pipeline():
    if pipeline_instance is None:
        raise HTTPException(status_code=500, detail="Pipeline engine not initialized.")
    return pipeline_instance

@router.get("/photos/{photo_path:path}")
async def serve_employee_photo(photo_path: str):
    """
    Serves employee reference photos strictly from exact filename on disk:
    1. Exact match in data/enrolled_photos (persistent volume)
    2. Exact match in Employees_Photo (pre-seeded assets)
    3. 404 if not found (strictly no fuzzy matching to avoid showing the wrong person)
    """
    import os
    import urllib.parse
    from fastapi.responses import FileResponse

    cleaned_path = urllib.parse.unquote(photo_path)
    base_name = os.path.basename(cleaned_path)

    # Priority 1: Persistent volume for newly enrolled photos
    persisted_path = os.path.join("data/enrolled_photos", base_name)
    if os.path.isfile(persisted_path):
        return FileResponse(persisted_path, media_type="image/jpeg")

    # Priority 2: Pre-seeded employee photos
    seeded_path = os.path.join("Employees_Photo", base_name)
    if os.path.isfile(seeded_path):
        return FileResponse(seeded_path, media_type="image/jpeg")

    raise HTTPException(status_code=404, detail="Employee photo not found.")

@router.get("/api/status")
async def get_system_status():
    pipeline = get_pipeline()
    total_enrolled = pipeline.gallery.total_vectors
    present_set = pipeline.get_present_employees_set()
    present_count = len(present_set)
    absent_count = max(0, total_enrolled - present_count)
    unknown_count = pipeline.get_unknown_count()
    return JSONResponse(
        content={
            "status": "ONLINE",
            "device": pipeline.config.get("hardware", {}).get("device", "cuda"),
            "total_enrolled": total_enrolled,
            "present_count": present_count,
            "absent_count": absent_count,
            "unknown_count": unknown_count,
            "active_mode": pipeline.active_mode,
        },
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )

@router.post("/api/record_unknown")
async def record_unknown_event(request: Request):
    """
    Critical Flag: Records an unverified / unknown person who failed 5 recognition attempts.
    """
    import base64
    pipeline = get_pipeline()
    crop_bytes = None
    full_frame_bytes = None

    try:
        body = await request.json()
        crop_data = body.get("crop_base64")
        if crop_data:
            if "," in crop_data:
                crop_data = crop_data.split(",", 1)[1]
            crop_bytes = base64.b64decode(crop_data)

        full_data = body.get("full_frame_base64")
        if full_data:
            if "," in full_data:
                full_data = full_data.split(",", 1)[1]
            full_frame_bytes = base64.b64decode(full_data)
    except Exception:
        pass

    res = pipeline.record_unknown_person(crop_bytes=crop_bytes, full_frame_bytes=full_frame_bytes)
    return JSONResponse(
        content=res,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )

@router.get("/api/mode")
async def get_mode():
    pipeline = get_pipeline()
    return {"mode": pipeline.active_mode}

@router.post("/api/mode")
async def set_mode(payload: Dict[str, Any]):
    pipeline = get_pipeline()
    new_mode = str(payload.get("mode", "ENTRY")).upper()
    if new_mode in ["ENTRY", "EXIT"]:
        pipeline.set_mode(new_mode)
        print(f"System operation mode updated to: {pipeline.active_mode} and cooldowns cleared for instant mode transition.")
        return {"status": "SUCCESS", "mode": pipeline.active_mode}
    raise HTTPException(status_code=400, detail="Invalid mode. Must be ENTRY or EXIT.")

@router.get("/api/attendance/recent")
async def get_recent_attendance():
    pipeline = get_pipeline()
    records = pipeline.db_repo.get_recent_attendance(limit=500) if pipeline.db_repo else []
    return JSONResponse(
        content={
            "attendance_records": records,
            "events": records,
            "active_mode": pipeline.active_mode
        },
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )

@router.post("/api/attendance/flush")
@router.post("/api/attendance/clear")
async def flush_attendance_records():
    """
    Flushes all attendance records from the database and resets live presence states.
    """
    pipeline = get_pipeline()
    res = pipeline.flush_attendance_events()
    return JSONResponse(
        content=res,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )
@router.post("/api/process_frame")
async def process_browser_frame(request: Request):
    """
    Browser-Camera Live Mode endpoint.
    Accepts raw JPEG bytes captured by the browser webcam (getUserMedia),
    runs the full recognition pipeline on it, and returns the annotated JPEG.
    Fast direct binary body — no python-multipart dependency needed.
    """
    pipeline = get_pipeline()

    # Read raw body bytes
    raw_bytes = await request.body()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Empty frame payload.")

    jpg_array = np.frombuffer(raw_bytes, dtype=np.uint8)
    bgr_frame = cv2.imdecode(jpg_array, cv2.IMREAD_COLOR)

    if bgr_frame is None:
        raise HTTPException(status_code=400, detail="Could not decode frame image.")

    # Run through the existing recognition pipeline
    result = pipeline.process_frame(bgr_frame)

    # Encode the annotated frame back to JPEG and return it
    _, jpeg_out = cv2.imencode('.jpg', result.annotated_frame,
                               [int(cv2.IMWRITE_JPEG_QUALITY), 65])
    return Response(content=jpeg_out.tobytes(), media_type="image/jpeg")

@router.post("/api/process_crop")
@router.post("/api/recognize_crop")
async def process_face_crop(request: Request):
    """
    Client-Side Face Detection Mode endpoint.
    Accepts a raw JPEG face crop captured + cropped by client-side face tracker.
    Runs AdaFace 512-d GPU embedding extraction + FAISS vector search, and records
    attendance if eligible (enforcing 10-second re-entry cooldown).
    """
    import base64
    pipeline = get_pipeline()
    content_type = request.headers.get("content-type", "").lower()

    crop_bytes = None
    full_frame_bytes = None
    kps = None

    if "application/json" in content_type:
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON payload.")

        crop_data = body.get("crop_base64") or body.get("crop") or body.get("image_base64")
        if not crop_data:
            raise HTTPException(status_code=400, detail="Cropped face image is required.")

        if "," in crop_data:
            crop_data = crop_data.split(",", 1)[1]

        try:
            crop_bytes = base64.b64decode(crop_data)
        except Exception:
            raise HTTPException(status_code=400, detail="Failed to decode crop base64 data.")

        full_data = body.get("full_frame_base64")
        if full_data:
            if "," in full_data:
                full_data = full_data.split(",", 1)[1]
            try:
                full_frame_bytes = base64.b64decode(full_data)
            except Exception:
                pass

        kps = body.get("kps") or body.get("landmarks")
    else:
        crop_bytes = await request.body()
        if not crop_bytes:
            raise HTTPException(status_code=400, detail="Crop binary payload is empty.")

    try:
        res = pipeline.process_crop(
            crop_bytes=crop_bytes,
            full_frame_bytes=full_frame_bytes,
            kps=kps
        )
        return res
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Face recognition failed: {str(exc)}")

@router.post("/api/enroll")
async def enroll_new_employee(request: Request):
    """
    Enrolls a new employee with photo, name, and employee ID.
    Accepts JSON with image_base64 / image_data or multipart form data.
    Runs face detection, alignment, embedding extraction, and updates FAISS gallery.
    """
    import base64
    pipeline = get_pipeline()
    content_type = request.headers.get("content-type", "").lower()

    name = ""
    employee_id = ""
    department = "General"
    image_bytes = None

    if "application/json" in content_type:
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON payload.")
        
        name = str(body.get("name", "")).strip()
        employee_id = str(body.get("employee_id", "")).strip()
        department = str(body.get("department", "General")).strip()
        img_data = body.get("image_base64") or body.get("image_data") or body.get("photo")

        if not img_data:
            raise HTTPException(status_code=400, detail="Photo image is required.")

        if "," in img_data:
            img_data = img_data.split(",", 1)[1]

        try:
            image_bytes = base64.b64decode(img_data)
        except Exception:
            raise HTTPException(status_code=400, detail="Failed to decode base64 image data.")
    else:
        try:
            form = await request.form()
            name = str(form.get("name", "")).strip()
            employee_id = str(form.get("employee_id", "")).strip()
            department = str(form.get("department", "General")).strip()
            file = form.get("photo") or form.get("file")
            if file and hasattr(file, "read"):
                image_bytes = await file.read()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read form upload: {e}")

    if not name:
        raise HTTPException(status_code=400, detail="Employee Name is required.")
    if not employee_id:
        raise HTTPException(status_code=400, detail="Employee ID is required.")
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Valid photo file is required.")

    try:
        res = pipeline.enroll_employee(
            name=name,
            employee_id=employee_id,
            image_input=image_bytes,
            department=department
        )
        return res
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Enrollment failed: {str(exc)}")

@router.get("/api/employees")
async def get_all_enrolled_employees():
    """
    Returns the complete directory of registered employees with real-time presence indicators.
    """
    pipeline = get_pipeline()
    employees = pipeline.get_enrolled_employees()
    return JSONResponse(
        content={
            "status": "SUCCESS",
            "total_enrolled": len(employees),
            "employees": employees
        },
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )

@router.delete("/api/employees/{employee_id}")
async def delete_enrolled_employee(employee_id: str):
    """
    Deletes an employee from FAISS gallery, in-memory cache, database, and disk.
    """
    pipeline = get_pipeline()
    try:
        res = pipeline.remove_employee(employee_id)
        return res
    except ValueError as val_err:
        raise HTTPException(status_code=404, detail=str(val_err))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to remove employee: {str(exc)}")

@router.post("/api/employees/remove")
async def post_remove_enrolled_employee(request: Request):
    """
    Alternative POST endpoint for removing an employee.
    """
    pipeline = get_pipeline()
    try:
        body = await request.json()
        employee_id = str(body.get("employee_id", "")).strip()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.")

    if not employee_id:
        raise HTTPException(status_code=400, detail="Employee ID is required.")

    try:
        res = pipeline.remove_employee(employee_id)
        return res
    except ValueError as val_err:
        raise HTTPException(status_code=404, detail=str(val_err))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to remove employee: {str(exc)}")

