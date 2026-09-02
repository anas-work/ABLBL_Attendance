import React, { useState, useEffect } from 'react';
import { X, Search, Users, User, Trash2, CheckCircle, AlertTriangle } from 'lucide-react';
import { fetchEmployees, deleteEmployee } from '../services/api';

export default function EmployeeModal({ isOpen, onClose, onUpdated }) {
  const [employees, setEmployees] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [deletingId, setDeletingId] = useState(null);
  const [confirmDeleteEmp, setConfirmDeleteEmp] = useState(null);
  const [feedbackMsg, setFeedbackMsg] = useState(null);

  const loadData = (showLoading = false) => {
    if (showLoading) setIsLoading(true);
    fetchEmployees()
      .then((data) => {
        setEmployees(data.employees || []);
        if (showLoading) setIsLoading(false);
      })
      .catch((err) => {
        console.error('Failed to load employees:', err);
        if (showLoading) setIsLoading(false);
      });
  };

  useEffect(() => {
    if (isOpen) {
      loadData(true);
      setConfirmDeleteEmp(null);
      setFeedbackMsg(null);

      // Auto-refresh employee list every 1.5s for real-time cross-device sync
      const pollInterval = setInterval(() => {
        loadData(false);
      }, 1500);

      return () => clearInterval(pollInterval);
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const handleDelete = async (emp) => {
    setDeletingId(emp.employee_id);
    try {
      await deleteEmployee(emp.employee_id);
      setFeedbackMsg({ type: 'success', text: `Removed ${emp.name} (${emp.employee_id}) successfully.` });
      setConfirmDeleteEmp(null);
      loadData();
      if (onUpdated) onUpdated();
    } catch (err) {
      setFeedbackMsg({ type: 'error', text: err.message || 'Failed to delete employee.' });
    } finally {
      setDeletingId(null);
    }
  };

  const filtered = employees.filter((emp) => {
    const q = searchTerm.toLowerCase();
    return (
      (emp.name && emp.name.toLowerCase().includes(q)) ||
      (emp.employee_id && emp.employee_id.toLowerCase().includes(q))
    );
  });

  const presentCount = employees.filter((e) => e.is_present).length;

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-container modal-large" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className="modal-title-group">
            <Users size={20} />
            <h2>Enrolled Employee Directory</h2>
            <span className="count-pill">{employees.length} Total</span>
            <span className="count-pill" style={{ background: 'rgba(16, 185, 129, 0.2)', color: '#34d399', borderColor: 'rgba(16, 185, 129, 0.4)' }}>
              {presentCount} Present
            </span>
          </div>
          <button className="close-btn" onClick={onClose}>
            <X size={18} />
          </button>
        </div>

        {feedbackMsg && (
          <div className={`alert-banner ${feedbackMsg.type === 'success' ? 'alert-success' : 'alert-error'}`} style={{ margin: '12px 20px 0 20px' }}>
            {feedbackMsg.type === 'success' ? <CheckCircle size={16} /> : <AlertTriangle size={16} />}
            <span>{feedbackMsg.text}</span>
            <button className="close-btn" style={{ marginLeft: 'auto', padding: 2 }} onClick={() => setFeedbackMsg(null)}>
              <X size={14} />
            </button>
          </div>
        )}

        <div className="modal-search-bar">
          <Search size={16} />
          <input
            type="text"
            placeholder="Search by employee name or ID..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            autoFocus
          />
        </div>

        <div className="modal-body modal-grid-scroll">
          {isLoading ? (
            <div className="modal-loading">Loading employee gallery...</div>
          ) : filtered.length === 0 ? (
            <div className="modal-empty">No matching employees found.</div>
          ) : (
            <div className="employee-grid">
              {filtered.map((emp) => {
                const isPresent = emp.is_present;
                const isPendingDelete = confirmDeleteEmp && confirmDeleteEmp.employee_id === emp.employee_id;

                return (
                  <div key={emp.employee_id || emp.name} className={`employee-card ${isPresent ? 'card-present' : ''}`}>
                    <div className="emp-photo-wrapper">
                      {emp.photo_url ? (
                        <img
                          src={emp.photo_url}
                          alt={emp.name}
                          className="emp-photo-img"
                          onError={(e) => { e.target.style.display = 'none'; }}
                        />
                      ) : (
                        <div className="emp-photo-placeholder">
                          <User size={28} />
                        </div>
                      )}
                    </div>

                    <div className="emp-info">
                      <div className="emp-name">{emp.name}</div>
                      <div className="emp-id">{emp.employee_id}</div>
                      
                      {/* Attendance Presence Status */}
                      <div className="emp-status-badge">
                        {isPresent ? (
                          <span className="status-pill status-present">
                            <span className="status-dot-green"></span> PRESENT
                          </span>
                        ) : (
                          <span className="status-pill status-absent">
                            <span className="status-dot-gray"></span> NOT CHECKED IN
                          </span>
                        )}
                      </div>
                    </div>

                    {/* Delete Employee Action */}
                    <div className="emp-actions">
                      {isPendingDelete ? (
                        <div className="delete-confirm-box" onClick={(e) => e.stopPropagation()}>
                          <span className="confirm-text">Delete?</span>
                          <button
                            className="btn-confirm-yes"
                            disabled={deletingId === emp.employee_id}
                            onClick={() => handleDelete(emp)}
                          >
                            {deletingId === emp.employee_id ? '...' : 'Yes'}
                          </button>
                          <button
                            className="btn-confirm-no"
                            onClick={() => setConfirmDeleteEmp(null)}
                          >
                            No
                          </button>
                        </div>
                      ) : (
                        <button
                          className="emp-delete-btn"
                          title={`Delete ${emp.name}`}
                          onClick={(e) => {
                            e.stopPropagation();
                            setConfirmDeleteEmp(emp);
                          }}
                        >
                          <Trash2 size={15} />
                        </button>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
