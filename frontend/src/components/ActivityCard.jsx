import React, { useState } from 'react';
import { User, AlertTriangle, ShieldAlert } from 'lucide-react';

export default function ActivityCard({ group, singleEvent, isGrouped = false, onSelect }) {
  // If Grouped Employee Mode (ALL FEED)
  if (isGrouped && group) {
    const isUnknownGroup = group.employee_id.includes('UNKNOWN') || group.employee_id.toLowerCase().includes('unknown');
    const availableTypes = Object.keys(group.eventsMap || {});
    const defaultType = group.eventsMap['UNKNOWN'] ? 'UNKNOWN' : (group.eventsMap['CHECK_IN'] ? 'CHECK_IN' : (group.eventsMap['RE_ENTRY'] ? 'RE_ENTRY' : availableTypes[0]));
    const [selectedType, setSelectedType] = useState(defaultType);

    const activeRec = group.eventsMap[selectedType] || group.eventsMap[defaultType] || {};
    const isUnknown = selectedType === 'UNKNOWN' || isUnknownGroup;
    const isExit = selectedType === 'CHECK_OUT';
    const isReEntry = selectedType === 'RE_ENTRY';

    let badgeColor = '#10b981'; // Green
    let frameLabel = 'CHECK-IN CAPTURE';

    if (isUnknown) {
      badgeColor = '#ef4444'; // Red
      frameLabel = 'UNKNOWN CAPTURE';
    } else if (isExit) {
      badgeColor = '#d946ef'; // Purple
      frameLabel = 'CHECK-OUT CAPTURE';
    } else if (isReEntry) {
      badgeColor = '#f59e0b'; // Orange
      frameLabel = 'RE-ENTRY CAPTURE';
    }

    const borderStyle = `1.5px solid ${badgeColor}`;

    return (
      <div
        className={`activity-card card-grouped ${isUnknown ? 'card-unknown' : ''}`}
        onClick={() => onSelect(activeRec)}
        title="Tap to view full details"
      >
        <div className="card-top">
          <div className="card-identity">
            {isUnknown && <ShieldAlert size={16} color="#ef4444" style={{ marginRight: 4 }} />}
            <span className="card-name" style={{ color: badgeColor }}>
              {isUnknown ? 'Unknown Person' : group.employee_id}
            </span>
          </div>

          <div className="sub-btn-group" onClick={(e) => e.stopPropagation()}>
            {availableTypes.map((type) => {
              const isActive = type === selectedType;
              let btnClass = 'pill-btn';
              if (isActive) {
                btnClass += type === 'UNKNOWN' ? ' active-unknown' : (type === 'CHECK_IN' ? ' active-checkin' : (type === 'RE_ENTRY' ? ' active-reentry' : ' active-checkout'));
              }
              const labelText = type === 'UNKNOWN' ? 'CRITICAL UNKNOWN' : (type === 'CHECK_IN' ? 'CHECK-IN' : (type === 'RE_ENTRY' ? 'RE-ENTRY' : 'CHECK-OUT'));
              return (
                <button
                  key={type}
                  className={btnClass}
                  onClick={() => setSelectedType(type)}
                >
                  {labelText}
                </button>
              );
            })}
          </div>
        </div>

        <div className="card-dual-photos">
          {/* Left: Selected Sub-Event Capture */}
          <div className="photo-column">
            <div className="photo-frame capture-frame" style={{ border: borderStyle }}>
              {activeRec.captured_frame_path ? (
                <img
                  src={activeRec.captured_frame_path}
                  alt="Capture"
                  className="photo-img"
                  onError={(e) => { e.target.style.display = 'none'; }}
                />
              ) : (
                <div className="photo-placeholder">
                  {isUnknown ? <AlertTriangle size={24} color="#ef4444" /> : <User size={24} />}
                </div>
              )}
            </div>
            <span className="photo-label" style={{ color: badgeColor }}>{frameLabel}</span>
          </div>

          {/* Right: Official Enrolled Photo or Unenrolled Warning */}
          <div className="photo-column">
            <div className="photo-frame enrolled-frame" style={{ border: isUnknown ? '1.5px dashed #ef4444' : '1.5px solid #3b82f6' }}>
              {group.enrolled_photo_path ? (
                <img
                  src={group.enrolled_photo_path}
                  alt="Official Enrolled"
                  className="photo-img"
                  onError={(e) => { e.target.style.display = 'none'; }}
                />
              ) : (
                <div className="photo-placeholder" style={{ background: isUnknown ? 'rgba(239, 68, 68, 0.1)' : undefined }}>
                  {isUnknown ? (
                    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
                      <AlertTriangle size={22} color="#ef4444" />
                      <span style={{ fontSize: 9, color: '#f87171', fontWeight: 700 }}>UNREGISTERED</span>
                    </div>
                  ) : (
                    <User size={24} />
                  )}
                </div>
              )}
            </div>
            <span className="photo-label" style={{ color: isUnknown ? '#f87171' : '#60a5fa' }}>
              {isUnknown ? 'NOT ENROLLED' : 'OFFICIAL PHOTO'}
            </span>
          </div>
        </div>
      </div>
    );
  }

  // Single Chronological Event Mode (CHECK-IN / CHECK-OUT / UNKNOWN tabs)
  const event = singleEvent || {};
  const isUnknown = event.event_type === 'UNKNOWN' || (event.employee_id && event.employee_id.includes('UNKNOWN'));
  const isExit = event.event_type === 'CHECK_OUT';
  const isReEntry = event.event_type === 'RE_ENTRY';

  let badgeClass = 'badge-check-in';
  let badgeLabel = 'CHECK-IN';
  let badgeColor = '#10b981';

  if (isUnknown) {
    badgeClass = 'badge-unknown';
    badgeLabel = 'CRITICAL: UNKNOWN';
    badgeColor = '#ef4444';
  } else if (isExit) {
    badgeClass = 'badge-check-out';
    badgeLabel = 'CHECK-OUT';
    badgeColor = '#d946ef';
  } else if (isReEntry) {
    badgeClass = 'badge-re-entry';
    badgeLabel = 'RE-ENTRY';
    badgeColor = '#f59e0b';
  }

  const timeStr = event.time_str || (event.timestamp ? new Date(event.timestamp).toLocaleTimeString() : 'Just now');

  return (
    <div
      className={`activity-card ${isUnknown ? 'card-unknown' : (isExit ? 'card-exit' : isReEntry ? 'card-re-entry' : 'card-entry')}`}
      onClick={() => onSelect(event)}
      title="Tap to view full details"
    >
      <div className="card-top">
        <div className="card-identity" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          {isUnknown && <ShieldAlert size={16} color="#ef4444" />}
          <span className="card-name" style={{ color: isUnknown ? '#f87171' : undefined }}>
            {isUnknown ? 'Unknown Person Detected' : (event.employee_name || event.employee_id)}
          </span>
        </div>
        <div className="card-meta">
          <span className={`event-badge ${badgeClass}`}>{badgeLabel}</span>
          <span className="event-time">{timeStr}</span>
        </div>
      </div>

      <div className="card-dual-photos">
        {/* Left: Live Capture Snapshot */}
        <div className="photo-column">
          <div className="photo-frame capture-frame" style={{ border: `1.5px solid ${badgeColor}` }}>
            {event.captured_frame_path ? (
              <img
                src={event.captured_frame_path}
                alt="Capture"
                className="photo-img"
                onError={(e) => { e.target.style.display = 'none'; }}
              />
            ) : (
              <div className="photo-placeholder">
                {isUnknown ? <AlertTriangle size={24} color="#ef4444" /> : <User size={24} />}
              </div>
            )}
          </div>
          <span className="photo-label" style={{ color: badgeColor }}>
            {isUnknown ? 'UNVERIFIED CAPTURE' : `${badgeLabel} CAPTURE`}
          </span>
        </div>

        {/* Right: Official Enrolled Photo or Unregistered Warning */}
        <div className="photo-column">
          <div className="photo-frame enrolled-frame" style={{ border: isUnknown ? '1.5px dashed #ef4444' : '1.5px solid #3b82f6' }}>
            {event.enrolled_photo_path ? (
              <img
                src={event.enrolled_photo_path}
                alt="Official Enrolled"
                className="photo-img"
                onError={(e) => { e.target.style.display = 'none'; }}
              />
            ) : (
              <div className="photo-placeholder" style={{ background: isUnknown ? 'rgba(239, 68, 68, 0.1)' : undefined }}>
                {isUnknown ? (
                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
                    <AlertTriangle size={22} color="#ef4444" />
                    <span style={{ fontSize: 9, color: '#f87171', fontWeight: 700 }}>UNREGISTERED</span>
                  </div>
                ) : (
                  <User size={24} />
                )}
              </div>
            )}
          </div>
          <span className="photo-label" style={{ color: isUnknown ? '#f87171' : '#60a5fa' }}>
            {isUnknown ? 'NOT ENROLLED' : 'OFFICIAL PHOTO'}
          </span>
        </div>
      </div>
    </div>
  );
}
