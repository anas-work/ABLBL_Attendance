import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import declarative_base, relationship

IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30), name="IST")

def get_ist_now():
    return datetime.datetime.now(IST)

Base = declarative_base()

class EmployeeModel(Base):
    __tablename__ = "employees"

    id = Column(Integer, primary_key=True, autoincrement=True)
    employee_id = Column(String(64), unique=True, nullable=False, index=True)
    name = Column(String(128), nullable=False)
    department = Column(String(64), default="General")
    status = Column(String(32), default="ACTIVE")
    created_at = Column(DateTime, default=get_ist_now)
    updated_at = Column(DateTime, default=get_ist_now, onupdate=get_ist_now)

    enrollments = relationship("EnrollmentModel", back_populates="employee")

class EnrollmentModel(Base):
    __tablename__ = "enrollments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    employee_id = Column(String(64), ForeignKey("employees.employee_id"), nullable=False)
    image_path = Column(String(256), nullable=False)
    embedding_identifier = Column(String(128), nullable=False)
    quality_score = Column(Float, default=1.0)
    created_at = Column(DateTime, default=get_ist_now)

    employee = relationship("EmployeeModel", back_populates="enrollments")

class CameraModel(Base):
    __tablename__ = "cameras"

    id = Column(String(64), primary_key=True)
    name = Column(String(128), nullable=False)
    source = Column(String(256), nullable=False)
    location = Column(String(128), default="Entrance")
    status = Column(String(32), default="ONLINE")
    created_at = Column(DateTime, default=get_ist_now)

class RecognitionEventModel(Base):
    __tablename__ = "recognition_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(String(64), nullable=False)
    employee_id = Column(String(64), nullable=True)
    track_id = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=get_ist_now, index=True)
    similarity = Column(Float, default=0.0)
    quality_score = Column(Float, default=0.0)
    decision = Column(String(32), nullable=False)

class AttendanceEventModel(Base):
    __tablename__ = "attendance_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    employee_id = Column(String(64), nullable=False, index=True)
    camera_id = Column(String(64), nullable=False)
    timestamp = Column(DateTime, default=get_ist_now, index=True)
    event_type = Column(String(32), default="CHECK_IN")
    captured_frame_path = Column(String(256), nullable=True)
    enrolled_photo_path = Column(String(256), nullable=True)
