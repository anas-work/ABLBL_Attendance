import React, { useState } from 'react';
import { Layers, ShieldAlert, Pause, Play, RefreshCw, AlertTriangle, CheckCircle2 } from 'lucide-react';
import ActivityCard from './ActivityCard';

export default function ActivityFeed({
  events = [],
  totalEnrolled = 132,
  presentCount = 0,
  absentCount = 132,
  unknownCount = 0,
  systemMode = 'ENTRY',
  isPaused = false,
  isFlushing = false,
  onTogglePause,
  onFlushFeed,
  onOpenEmployees,
  onSelectEvent
}) {
  const [activeTab, setActiveTab] = useState('ALL');
  const [showFlushConfirm, setShowFlushConfirm] = useState(false);
  const [flushNotice, setFlushNotice] = useState(null);

  // Count unknown events in list
  const unknownList = events.filter((ev) => ev.event_type === 'UNKNOWN' || (ev.employee_id && ev.employee_id.includes('UNKNOWN')));
  const effectiveUnknownCount = Math.max(unknownCount, unknownList.length);

  // Group events for ALL FEED tab:
  // - Enrolled employees are grouped by employee_id with toggle pills (CHECK-IN / CHECK-OUT / RE-ENTRY)
  // - Each UNKNOWN person incident is preserved as its own distinct card
  const getGroupedEmployeeRecords = () => {
    const list = [];
    const empMap = new Map();

    for (const rec of events) {
      const isUnknown = rec.event_type === 'UNKNOWN' || (rec.employee_id && rec.employee_id.includes('UNKNOWN'));

      if (isUnknown) {
        // Distinct individual unknown detection entry
        list.push({
          key: `UNKNOWN_${rec.id || rec.timestamp}_${rec.captured_frame_path || Math.random()}`,
          employee_id: 'Unknown Person',
          enrolled_photo_path: null,
          isUnknown: true,
          eventsMap: { 'UNKNOWN': rec },
          singleEvent: rec
        });
      } else {
        // Enrolled employee grouped by ID
        if (!empMap.has(rec.employee_id)) {
          const groupObj = {
            key: rec.employee_id,
            employee_id: rec.employee_id,
            enrolled_photo_path: rec.enrolled_photo_path,
            isUnknown: false,
            eventsMap: {}
          };
          empMap.set(rec.employee_id, groupObj);
          list.push(groupObj);
        }
        const empGroup = empMap.get(rec.employee_id);
        if (!empGroup.enrolled_photo_path && rec.enrolled_photo_path) {
          empGroup.enrolled_photo_path = rec.enrolled_photo_path;
        }
        empGroup.eventsMap[rec.event_type] = rec;
      }
    }
    return list;
  };

  // Individual events for CHECK_IN / CHECK_OUT / UNKNOWN tabs
  const filteredSingleEvents = events.filter((ev) => {
    if (activeTab === 'CHECK_IN') {
      return ev.event_type === 'CHECK_IN' || ev.event_type === 'RE_ENTRY';
    }
    if (activeTab === 'CHECK_OUT') {
      return ev.event_type === 'CHECK_OUT';
    }
    if (activeTab === 'UNKNOWN') {
      return ev.event_type === 'UNKNOWN' || (ev.employee_id && ev.employee_id.includes('UNKNOWN'));
    }
    return true;
  });

  const groupedList = getGroupedEmployeeRecords();

  const handleExecuteFlush = async () => {
    setShowFlushConfirm(false);
    if (onFlushFeed) {
      await onFlushFeed();
      setFlushNotice('Feed flushed and reset fresh!');
      setTimeout(() => setFlushNotice(null), 3000);
    }
  };

  return (
    <aside className="activity-feed-panel">
      {/* Top Presence & Unknown Quick-Stats Row */}
      <div className="stats-row">
        <div className="stat-card" onClick={onOpenEmployees} style={{ cursor: 'pointer' }}>
          <div className="stat-header">
            <span className="stat-title">PRESENT</span>
            <span className="stat-link" style={{ color: '#34d399' }}>
              <span>ACTIVE</span>
            </span>
          </div>
          <div className="stat-value" style={{ color: '#34d399' }}>{presentCount}</div>
        </div>

        <div className="stat-card" onClick={onOpenEmployees} style={{ cursor: 'pointer' }}>
          <div className="stat-header">
            <span className="stat-title">ABSENT</span>
          </div>
          <div className="stat-value" style={{ color: '#94a3b8' }}>{absentCount}</div>
        </div>

        <div className="stat-card stat-card-critical" onClick={() => setActiveTab('UNKNOWN')} style={{ cursor: 'pointer' }}>
          <div className="stat-header">
            <span className="stat-title" style={{ color: '#f87171' }}>UNKNOWN</span>
            <span className="stat-link" style={{ color: '#ef4444' }}>
              <span>FLAGGED</span>
            </span>
          </div>
          <div className="stat-value" style={{ color: '#ef4444' }}>{effectiveUnknownCount}</div>
        </div>

        <div className="stat-card">
          <div className="stat-header">
            <span className="stat-title">MODE</span>
          </div>
          <div className={`stat-value-mode ${systemMode === 'EXIT' ? 'mode-exit-text' : 'mode-entry-text'}`}>
            {systemMode}
          </div>
        </div>
      </div>

      {/* Activity Feed Header with Stop & Flush Actions */}
      <div className="feed-header-section">
        <div className="feed-header-title">
          <Layers size={16} />
          <span>Activity Feed</span>
          {isPaused ? (
            <span className="feed-live-badge badge-paused">⏸️ PAUSED</span>
          ) : (
            <span className="feed-live-badge badge-live">● LIVE</span>
          )}
        </div>

        <div className="feed-controls-group">
          {/* Stop / Resume Button */}
          <button
            className={`feed-action-btn ${isPaused ? 'btn-resume-state' : 'btn-stop-state'}`}
            onClick={onTogglePause}
            title={isPaused ? "Resume real-time feed updates" : "Stop live feed updates from refreshing"}
          >
            {isPaused ? <Play size={13} /> : <Pause size={13} />}
            <span>{isPaused ? 'Resume' : 'Stop'}</span>
          </button>

          {/* Flush & Refresh Button */}
          <button
            className="feed-action-btn btn-flush-state"
            onClick={() => setShowFlushConfirm(true)}
            disabled={isFlushing}
            title="Flush old events and start a fresh feed"
          >
            <RefreshCw size={13} className={isFlushing ? 'spin-icon' : ''} />
            <span>Flush</span>
          </button>
        </div>
      </div>

      {/* Flush Confirmation Banner */}
      {showFlushConfirm && (
        <div className="feed-flush-confirm-box">
          <div className="flush-confirm-header">
            <AlertTriangle size={15} color="#fbbf24" />
            <span>Flush all feed records & start fresh?</span>
          </div>
          <div className="flush-confirm-actions">
            <button className="btn-confirm-flush" onClick={handleExecuteFlush}>
              Yes, Flush
            </button>
            <button className="btn-cancel-flush" onClick={() => setShowFlushConfirm(false)}>
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Paused Alert Banner */}
      {isPaused && (
        <div className="feed-paused-alert">
          <Pause size={14} />
          <span>Feed updates stopped. Real-time changes are frozen.</span>
        </div>
      )}

      {/* Flush Success Notice */}
      {flushNotice && (
        <div className="feed-flush-success">
          <CheckCircle2 size={14} />
          <span>{flushNotice}</span>
        </div>
      )}

      <div className="feed-tabs">
        <button
          className={`tab-btn ${activeTab === 'CHECK_IN' ? 'active-checkin' : ''}`}
          onClick={() => setActiveTab('CHECK_IN')}
        >
          CHECK-IN
        </button>
        <button
          className={`tab-btn ${activeTab === 'CHECK_OUT' ? 'active-checkout' : ''}`}
          onClick={() => setActiveTab('CHECK_OUT')}
        >
          CHECK-OUT
        </button>
        <button
          className={`tab-btn ${activeTab === 'UNKNOWN' ? 'active-unknown-tab' : ''}`}
          onClick={() => setActiveTab('UNKNOWN')}
        >
          <ShieldAlert size={13} style={{ marginRight: 3, verticalAlign: 'middle' }} />
          UNKNOWN {effectiveUnknownCount > 0 && `(${effectiveUnknownCount})`}
        </button>
        <button
          className={`tab-btn ${activeTab === 'ALL' ? 'active-all' : ''}`}
          onClick={() => setActiveTab('ALL')}
        >
          ALL FEED
        </button>
      </div>

      {/* Events List */}
      <div className="feed-list">
        {activeTab === 'ALL' ? (
          groupedList.length === 0 ? (
            <div className="feed-empty-state">
              <p>No recent activity records found.</p>
              <span>Events will appear automatically when faces are recognized.</span>
            </div>
          ) : (
            groupedList.map((grp) => (
              <ActivityCard
                key={grp.key || grp.employee_id}
                group={grp}
                singleEvent={grp.singleEvent}
                isGrouped={!grp.isUnknown}
                onSelect={onSelectEvent}
              />
            ))
          )
        ) : (
          filteredSingleEvents.length === 0 ? (
            <div className="feed-empty-state">
              {activeTab === 'UNKNOWN' ? (
                <div>
                  <ShieldAlert size={32} color="#64748b" style={{ margin: '0 auto 8px auto', display: 'block' }} />
                  <p>No unverified or unknown persons detected.</p>
                  <span>Critical alerts appear here after 3 seconds of unverified evaluation.</span>
                </div>
              ) : (
                <p>No matching {activeTab === 'CHECK_IN' ? 'check-in' : 'check-out'} events.</p>
              )}
            </div>
          ) : (
            filteredSingleEvents.map((ev) => (
              <ActivityCard
                key={ev.id || `${ev.employee_id}-${ev.timestamp}-${ev.captured_frame_path}`}
                singleEvent={ev}
                isGrouped={false}
                onSelect={onSelectEvent}
              />
            ))
          )
        )}
      </div>
    </aside>
  );
}
