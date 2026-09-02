import React from 'react';
import { X, CheckCircle2, ShieldCheck, Clock, User, Camera } from 'lucide-react';

export default function EventDetailModal({ event, onClose }) {
  if (!event) return null;

  const isExit = event.event_type === 'CHECK_OUT';
  const isReEntry = event.event_type === 'RE_ENTRY';
  const timeStr = event.timestamp ? new Date(event.timestamp).toLocaleString() : 'N/A';

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-container modal-medium" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className="modal-title-group">
            <ShieldCheck size={22} className="text-emerald-400" />
            <h2>Attendance Verification Detail</h2>
          </div>
          <button className="close-btn" onClick={onClose}>
            <X size={18} />
          </button>
        </div>

        <div className="modal-body">
          <div className="detail-meta-box">
            <div className="meta-item">
              <span className="meta-label">Employee</span>
              <span className="meta-value">{event.employee_name || event.employee_id}</span>
            </div>
            <div className="meta-item">
              <span className="meta-label">Event Type</span>
              <span className={`event-badge ${isExit ? 'badge-check-out' : isReEntry ? 'badge-re-entry' : 'badge-check-in'}`}>
                {event.event_type}
              </span>
            </div>
            <div className="meta-item">
              <span className="meta-label">Timestamp</span>
              <span className="meta-value">{timeStr}</span>
            </div>
            <div className="meta-item">
              <span className="meta-label">Camera</span>
              <span className="meta-value">{event.camera_id || 'CLIENT_DEVICE_CAM'}</span>
            </div>
          </div>

          <div className="detail-dual-comparison">
            <div className="comparison-box">
              <div className="comparison-title">
                <Camera size={14} />
                <span>Live Event Capture</span>
              </div>
              <div className="comparison-img-wrapper">
                {event.captured_frame_path ? (
                  <img src={event.captured_frame_path} alt="Live Snapshot" />
                ) : (
                  <div className="comparison-placeholder">No capture image</div>
                )}
              </div>
            </div>

            <div className="comparison-box">
              <div className="comparison-title">
                <User size={14} />
                <span>Enrolled Reference Photo</span>
              </div>
              <div className="comparison-img-wrapper">
                {event.enrolled_photo_path ? (
                  <img src={event.enrolled_photo_path} alt="Enrolled Reference" />
                ) : (
                  <div className="comparison-placeholder">No enrolled photo</div>
                )}
              </div>
            </div>
          </div>
        </div>

        <div className="modal-footer">
          <button className="btn-secondary" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
