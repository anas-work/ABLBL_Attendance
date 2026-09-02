/**
 * API Service for Backend Communication (Cross-Device Real-Time Sync)
 */

export const fetchStatus = async () => {
  const res = await fetch(`/api/status?_t=${Date.now()}`, {
    cache: 'no-store',
    headers: {
      'Cache-Control': 'no-cache',
      'Pragma': 'no-cache'
    }
  });
  if (!res.ok) throw new Error('Failed to fetch status');
  return res.json();
};

export const switchMode = async (mode) => {
  const res = await fetch('/api/mode', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ mode })
  });
  if (!res.ok) throw new Error('Failed to switch mode');
  return res.json();
};

export const fetchRecentAttendance = async (limit = 50) => {
  const res = await fetch(`/api/attendance/recent?limit=${limit}&_t=${Date.now()}`, {
    cache: 'no-store',
    headers: {
      'Cache-Control': 'no-cache',
      'Pragma': 'no-cache'
    }
  });
  if (!res.ok) throw new Error('Failed to fetch recent attendance');
  return res.json();
};

export const fetchEmployees = async () => {
  const res = await fetch(`/api/employees?_t=${Date.now()}`, {
    cache: 'no-store'
  });
  if (!res.ok) throw new Error('Failed to fetch employees');
  return res.json();
};

export const enrollEmployee = async (formData) => {
  const res = await fetch('/api/enroll', {
    method: 'POST',
    body: formData
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || 'Enrollment failed');
  }
  return res.json();
};

export const deleteEmployee = async (employeeId) => {
  const res = await fetch(`/api/employees/${encodeURIComponent(employeeId)}`, {
    method: 'DELETE'
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || 'Failed to delete employee');
  }
  return res.json();
};

export const flushAttendanceFeed = async () => {
  const res = await fetch('/api/attendance/flush', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' }
  });
  if (!res.ok) throw new Error('Failed to flush attendance feed');
  return res.json();
};
