import React from 'react';
import { Users, UserPlus, ShieldAlert } from 'lucide-react';

export default function Header({
  systemMode,
  totalEnrolled = 132,
  presentCount = 0,
  absentCount = 132,
  unknownCount = 0,
  onSwitchMode,
  onOpenEmployees,
  onOpenEnroll
}) {
  return (
    <header className="app-header">
      <div className="header-brand">
        <div className="brand-logo">
          <div className="pulse-indicator online" />
          <span className="brand-title">AI Monk Attendance</span>
        </div>
        <span className="brand-badge">EDGE AI PRO</span>
      </div>

      {/* Live Presence Attendance Counters in Header */}
      <div className="header-stats-group" onClick={onOpenEmployees} title="Click to view full employee directory">
        <div className="header-stat-pill stat-present-pill">
          <span className="header-stat-dot dot-present" />
          <span className="header-stat-label">PRESENT</span>
          <span className="header-stat-value">{presentCount}</span>
        </div>

        <div className="header-stat-pill stat-absent-pill">
          <span className="header-stat-dot dot-absent" />
          <span className="header-stat-label">ABSENT</span>
          <span className="header-stat-value">{absentCount}</span>
        </div>

        <div className="header-stat-pill stat-unknown-pill">
          <span className="header-stat-dot dot-unknown" />
          <span className="header-stat-label">UNKNOWN</span>
          <span className="header-stat-value">{unknownCount}</span>
        </div>

        <div className="header-stat-pill stat-total-pill">
          <span className="header-stat-label">TOTAL</span>
          <span className="header-stat-value">{totalEnrolled}</span>
        </div>
      </div>

      <div className="header-actions">
        <button className="nav-btn" onClick={onOpenEmployees}>
          <Users size={16} />
          <span>View Employees</span>
        </button>

        <button className="nav-btn btn-primary" onClick={onOpenEnroll}>
          <UserPlus size={16} />
          <span>Enroll Employee</span>
        </button>

        <div className="mode-toggle">
          <button
            className={`mode-btn ${systemMode === 'ENTRY' ? 'active-entry' : ''}`}
            onClick={() => onSwitchMode('ENTRY')}
          >
            <span>ENTRY MODE</span>
          </button>
          <button
            className={`mode-btn ${systemMode === 'EXIT' ? 'active-exit' : ''}`}
            onClick={() => onSwitchMode('EXIT')}
          >
            <span>EXIT MODE</span>
          </button>
        </div>
      </div>
    </header>
  );
}
