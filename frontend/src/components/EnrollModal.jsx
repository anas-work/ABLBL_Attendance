import React, { useState, useRef, useEffect, useCallback } from 'react';
import { X, UserPlus, Upload, Camera, CheckCircle2, AlertCircle, RefreshCw, RotateCcw, Image as ImageIcon } from 'lucide-react';
import { enrollEmployee } from '../services/api';

export default function EnrollModal({ isOpen, onClose, onEnrolled }) {
  const [name, setName] = useState('');
  const [employeeId, setEmployeeId] = useState('');
  const [captureMode, setCaptureMode] = useState('CAMERA'); // 'CAMERA' or 'UPLOAD'
  const [photoFile, setPhotoFile] = useState(null);
  const [photoPreview, setPhotoPreview] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [resultMsg, setResultMsg] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);
  const [cameraError, setCameraError] = useState(null);
  const [isCameraActive, setIsCameraActive] = useState(false);

  const fileInputRef = useRef(null);
  const videoRef = useRef(null);
  const streamRef = useRef(null);

  // Stop camera tracks cleanly
  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      try {
        streamRef.current.getTracks().forEach((track) => track.stop());
      } catch (e) {
        console.warn('Error stopping camera track:', e);
      }
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsCameraActive(false);
  }, []);

  // Start webcam stream
  const startCamera = useCallback(async () => {
    stopCamera();
    setCameraError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'user',
          width: { ideal: 640 },
          height: { ideal: 480 }
        },
        audio: false
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current.play().catch((err) => {
            console.warn('Video play prevented:', err);
          });
        };
      }
      setIsCameraActive(true);
    } catch (err) {
      console.warn('Webcam access error:', err);
      setCameraError('Unable to access webcam. Please ensure camera permissions are allowed, or switch to Upload Photo.');
      setIsCameraActive(false);
    }
  }, [stopCamera]);

  // Handle open/close and mode change lifecycle
  useEffect(() => {
    if (isOpen && captureMode === 'CAMERA' && !photoPreview) {
      startCamera();
    } else {
      stopCamera();
    }

    return () => {
      stopCamera();
    };
  }, [isOpen, captureMode, photoPreview, startCamera, stopCamera]);

  if (!isOpen) return null;

  const handleClose = () => {
    stopCamera();
    setName('');
    setEmployeeId('');
    setPhotoFile(null);
    setPhotoPreview(null);
    setErrorMsg(null);
    setResultMsg(null);
    setCameraError(null);
    onClose();
  };

  const handleFileChange = (e) => {
    const file = e.target.files && e.target.files[0];
    if (file) {
      setPhotoFile(file);
      const reader = new FileReader();
      reader.onload = (ev) => {
        setPhotoPreview(ev.target.result);
      };
      reader.readAsDataURL(file);
    }
  };

  // Synchronously snap photo from live video and convert to File object
  const handleSnapPhoto = () => {
    if (!videoRef.current) return;
    const video = videoRef.current;
    const w = video.videoWidth || 640;
    const h = video.videoHeight || 480;

    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    
    // Draw mirrored video stream correctly to canvas
    ctx.translate(w, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, w, h);

    const dataUrl = canvas.toDataURL('image/jpeg', 0.95);
    setPhotoPreview(dataUrl);

    // Convert dataURL to Blob & File synchronously
    try {
      const arr = dataUrl.split(',');
      const mime = arr[0].match(/:(.*?);/)[1];
      const bstr = atob(arr[1]);
      let n = bstr.length;
      const u8arr = new Uint8Array(n);
      while (n--) {
        u8arr[n] = bstr.charCodeAt(n);
      }
      const blob = new Blob([u8arr], { type: mime });
      const snapFileName = `enroll_${employeeId.trim() || 'camera'}_${Date.now()}.jpg`;
      const file = new File([blob], snapFileName, { type: 'image/jpeg' });
      setPhotoFile(file);
    } catch (err) {
      console.error('Error generating photo file from canvas:', err);
    }

    stopCamera();
  };

  const handleRetake = () => {
    setPhotoFile(null);
    setPhotoPreview(null);
    if (captureMode === 'CAMERA') {
      startCamera();
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setErrorMsg(null);
    setResultMsg(null);

    const cleanName = name.trim();
    const cleanId = employeeId.trim();

    if (!cleanName || !cleanId) {
      setErrorMsg('Please enter both Full Name and Employee ID.');
      return;
    }

    if (!photoFile && !photoPreview) {
      setErrorMsg('Please capture or upload a portrait photo before registering.');
      return;
    }

    setIsSubmitting(true);

    try {
      let finalFile = photoFile;
      // Fallback: If for any reason photoFile wasn't created but photoPreview dataURL exists
      if (!finalFile && photoPreview) {
        const arr = photoPreview.split(',');
        const mime = arr[0].match(/:(.*?);/)[1];
        const bstr = atob(arr[1]);
        let n = bstr.length;
        const u8arr = new Uint8Array(n);
        while (n--) {
          u8arr[n] = bstr.charCodeAt(n);
        }
        const blob = new Blob([u8arr], { type: mime });
        finalFile = new File([blob], `enroll_${cleanId}_${Date.now()}.jpg`, { type: 'image/jpeg' });
      }

      const formData = new FormData();
      formData.append('name', cleanName);
      formData.append('employee_id', cleanId);
      formData.append('photo', finalFile);

      const res = await enrollEmployee(formData);
      setIsSubmitting(false);
      setResultMsg(`Successfully enrolled ${res.name} (${res.employee_id})!`);

      if (onEnrolled) onEnrolled();

      setTimeout(() => {
        handleClose();
      }, 1200);
    } catch (err) {
      setIsSubmitting(false);
      setErrorMsg(err.message || 'Failed to enroll employee.');
    }
  };

  const isFormReady = Boolean(name.trim() && employeeId.trim() && (photoFile || photoPreview));

  return (
    <div className="modal-backdrop" onClick={handleClose}>
      <div className="modal-container modal-medium" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className="modal-title-group">
            <UserPlus size={20} />
            <h2>Enroll New Employee</h2>
          </div>
          <button className="close-btn" onClick={handleClose} type="button">
            <X size={18} />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="enroll-form">
          <div className="modal-body">
            {errorMsg && (
              <div className="alert-banner alert-error">
                <AlertCircle size={16} />
                <span>{errorMsg}</span>
              </div>
            )}

            {resultMsg && (
              <div className="alert-banner alert-success">
                <CheckCircle2 size={16} />
                <span>{resultMsg}</span>
              </div>
            )}

            <div className="form-group">
              <label>Full Name *</label>
              <input
                type="text"
                placeholder="e.g. Rahul Sharma"
                value={name}
                onChange={(e) => setName(e.target.value)}
                required
              />
            </div>

            <div className="form-group">
              <label>Employee ID *</label>
              <input
                type="text"
                placeholder="e.g. ABL1045"
                value={employeeId}
                onChange={(e) => setEmployeeId(e.target.value)}
                required
              />
            </div>

            {/* Mode Switcher: Live Webcam vs File Upload */}
            <div className="form-group">
              <label>Enrollment Photo *</label>
              <div className="enroll-mode-tabs">
                <button
                  type="button"
                  className={`enroll-tab-btn ${captureMode === 'CAMERA' ? 'active-enroll-tab' : ''}`}
                  onClick={() => {
                    setCaptureMode('CAMERA');
                    setPhotoFile(null);
                    setPhotoPreview(null);
                  }}
                >
                  <Camera size={14} />
                  <span>Live Webcam Capture</span>
                </button>
                <button
                  type="button"
                  className={`enroll-tab-btn ${captureMode === 'UPLOAD' ? 'active-enroll-tab' : ''}`}
                  onClick={() => {
                    setCaptureMode('UPLOAD');
                    setPhotoFile(null);
                    setPhotoPreview(null);
                    stopCamera();
                  }}
                >
                  <Upload size={14} />
                  <span>Upload Photo File</span>
                </button>
              </div>
            </div>

            {/* Photo Capture / Upload Box */}
            <div className="form-group">
              {photoPreview ? (
                /* Snapped / Uploaded Photo Preview Card */
                <div className="enroll-preview-box">
                  <div className="enroll-preview-img-wrapper">
                    <img src={photoPreview} alt="Profile Preview" />
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6 }}>
                    <span style={{ fontSize: '11.5px', color: '#34d399', fontWeight: 600 }}>
                      ✓ Photo Ready for Registration
                    </span>
                    <button type="button" className="btn-retake-photo" onClick={handleRetake}>
                      <RotateCcw size={13} />
                      <span>Retake / Change Photo</span>
                    </button>
                  </div>
                </div>
              ) : captureMode === 'CAMERA' ? (
                /* Live Webcam Capture Box */
                <div className="enroll-camera-viewport">
                  {cameraError ? (
                    <div className="enroll-camera-error">
                      <AlertCircle size={24} color="#ef4444" />
                      <p>{cameraError}</p>
                      <button
                        type="button"
                        className="btn-secondary"
                        onClick={() => setCaptureMode('UPLOAD')}
                        style={{ marginTop: 6 }}
                      >
                        Switch to Photo Upload
                      </button>
                    </div>
                  ) : (
                    <div className="camera-live-stream-box">
                      <video
                        ref={videoRef}
                        autoPlay
                        playsInline
                        muted
                        className="enroll-live-video"
                      />
                      <div className="face-guide-oval">
                        <span>Align Face Inside Oval</span>
                      </div>
                      <button
                        type="button"
                        className="btn-snap-capture"
                        onClick={handleSnapPhoto}
                      >
                        <Camera size={16} />
                        <span>Capture Face Photo</span>
                      </button>
                    </div>
                  )}
                </div>
              ) : (
                /* File Upload Dropzone */
                <div
                  className="photo-upload-dropzone"
                  onClick={() => fileInputRef.current && fileInputRef.current.click()}
                >
                  <input
                    type="file"
                    ref={fileInputRef}
                    style={{ display: 'none' }}
                    accept="image/jpeg,image/png,image/webp"
                    onChange={handleFileChange}
                  />
                  <div className="upload-placeholder">
                    <Upload size={28} />
                    <p>Click or drag a clear front-facing portrait photo</p>
                    <span>Supports JPEG, PNG, or WebP</span>
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="modal-footer">
            <button type="button" className="btn-secondary" onClick={handleClose}>
              Cancel
            </button>
            <button
              type="submit"
              className="btn-primary"
              disabled={isSubmitting || !isFormReady}
            >
              {isSubmitting ? (
                <>
                  <RefreshCw size={15} className="spin-icon" />
                  <span>Extracting Features & Registering...</span>
                </>
              ) : (
                <>
                  <UserPlus size={15} />
                  <span>Register Employee</span>
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
