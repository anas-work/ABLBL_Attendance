import React, { useState, useEffect, useCallback } from 'react';
import Header from './components/Header';
import VideoPlayer from './components/VideoPlayer';
import ActivityFeed from './components/ActivityFeed';
import EmployeeModal from './components/EmployeeModal';
import EnrollModal from './components/EnrollModal';
import EventDetailModal from './components/EventDetailModal';
import { fetchStatus, switchMode, fetchRecentAttendance, flushAttendanceFeed } from './services/api';
import './App.css';

export default function App() {
  const [systemMode, setSystemMode] = useState('ENTRY');
  const [events, setEvents] = useState([]);
  const [totalEnrolled, setTotalEnrolled] = useState(132);
  const [presentCount, setPresentCount] = useState(0);
  const [absentCount, setAbsentCount] = useState(132);
  const [unknownCount, setUnknownCount] = useState(0);
  const [isFeedPaused, setIsFeedPaused] = useState(false);
  const [isFlushing, setIsFlushing] = useState(false);

  // Modals
  const [isEmployeeModalOpen, setIsEmployeeModalOpen] = useState(false);
  const [isEnrollModalOpen, setIsEnrollModalOpen] = useState(false);
  const [selectedDetailEvent, setSelectedDetailEvent] = useState(null);

  // Load Status & Mode
  const loadStatus = useCallback(async () => {
    try {
      const statusData = await fetchStatus();
      if (statusData.active_mode) {
        setSystemMode(statusData.active_mode);
      }
      if (statusData.total_enrolled !== undefined) {
        setTotalEnrolled(statusData.total_enrolled);
      }
      if (statusData.present_count !== undefined) {
        setPresentCount(statusData.present_count);
      }
      if (statusData.absent_count !== undefined) {
        setAbsentCount(statusData.absent_count);
      }
      if (statusData.unknown_count !== undefined) {
        setUnknownCount(statusData.unknown_count);
      }
    } catch (err) {
      console.warn('Status poll warning:', err);
    }
  }, []);

  // Load Recent Attendance Events
  const loadRecentEvents = useCallback(async () => {
    try {
      const data = await fetchRecentAttendance(50);
      const list = data.attendance_records || data.events || [];
      setEvents(list);
    } catch (err) {
      console.warn('Attendance poll warning:', err);
    }
  }, []);

  useEffect(() => {
    loadStatus();
    loadRecentEvents();

    if (isFeedPaused) return;

    const interval = setInterval(() => {
      loadRecentEvents();
      loadStatus();
    }, 1500);

    return () => clearInterval(interval);
  }, [loadStatus, loadRecentEvents, isFeedPaused]);

  const handleSwitchMode = async (newMode) => {
    try {
      const res = await switchMode(newMode);
      setSystemMode(res.mode || newMode);
    } catch (err) {
      console.error('Failed to switch mode:', err);
    }
  };

  const handleTogglePauseFeed = () => {
    setIsFeedPaused((prev) => !prev);
  };

  const handleFlushFeed = async () => {
    try {
      setIsFlushing(true);
      await flushAttendanceFeed();
      setEvents([]);
      setPresentCount(0);
      setUnknownCount(0);
      await loadStatus();
      await loadRecentEvents();
    } catch (err) {
      console.error('Failed to flush attendance feed:', err);
    } finally {
      setIsFlushing(false);
    }
  };

  return (
    <div className="app-root">
      <Header
        systemMode={systemMode}
        totalEnrolled={totalEnrolled}
        presentCount={presentCount}
        absentCount={absentCount}
        unknownCount={unknownCount}
        onSwitchMode={handleSwitchMode}
        onOpenEmployees={() => setIsEmployeeModalOpen(true)}
        onOpenEnroll={() => setIsEnrollModalOpen(true)}
      />

      <main className="main-layout">
        <VideoPlayer
          systemMode={systemMode}
          onAttendanceEvent={() => {
            if (!isFeedPaused) {
              loadRecentEvents();
              loadStatus();
            }
          }}
        />

        <ActivityFeed
          events={events}
          totalEnrolled={totalEnrolled}
          presentCount={presentCount}
          absentCount={absentCount}
          unknownCount={unknownCount}
          systemMode={systemMode}
          isPaused={isFeedPaused}
          isFlushing={isFlushing}
          onTogglePause={handleTogglePauseFeed}
          onFlushFeed={handleFlushFeed}
          onOpenEmployees={() => setIsEmployeeModalOpen(true)}
          onSelectEvent={(ev) => setSelectedDetailEvent(ev)}
        />
      </main>

      {/* Modals */}
      <EmployeeModal
        isOpen={isEmployeeModalOpen}
        onClose={() => setIsEmployeeModalOpen(false)}
      />

      <EnrollModal
        isOpen={isEnrollModalOpen}
        onClose={() => setIsEnrollModalOpen(false)}
        onEnrolled={() => {
          loadStatus();
          loadRecentEvents();
        }}
      />

      <EventDetailModal
        event={selectedDetailEvent}
        onClose={() => setSelectedDetailEvent(null)}
      />
    </div>
  );
}
