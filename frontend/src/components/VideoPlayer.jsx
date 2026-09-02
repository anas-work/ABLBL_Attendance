import React, { useRef, useEffect, useState, useCallback } from 'react';
import { Camera, RefreshCw, Pause, AlertCircle } from 'lucide-react';
import { UltraLightDetector } from '../engine/UltraLightDetector';
import { ClientIoUTracker } from '../engine/ClientIoUTracker';
import { CanvasRenderer } from '../engine/CanvasRenderer';
import { CropDispatcher } from '../engine/CropDispatcher';

export default function VideoPlayer({ systemMode, onAttendanceEvent }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const [isCameraActive, setIsCameraActive] = useState(false);
  const [facingMode, setFacingMode] = useState('user');
  const [cameraError, setCameraError] = useState(null);
  const [isStartingEngine, setIsStartingEngine] = useState(false);

  const detectorRef = useRef(null);
  const trackerRef = useRef(null);
  const rendererRef = useRef(null);
  const dispatcherRef = useRef(null);
  const streamRef = useRef(null);
  const activePopupRef = useRef(null);
  const animFrameRef = useRef(null);

  const telemetryRef = useRef({
    fps: 30.0,
    detectMs: 8.0,
    trackMs: 0.1,
    e2eMs: 8.2,
    frameCount: 0,
    lastFpsTime: performance.now(),
  });

  // Initialize Engines
  useEffect(() => {
    detectorRef.current = new UltraLightDetector({ confThreshold: 0.58, nmsThreshold: 0.25 });
    trackerRef.current = new ClientIoUTracker(0.12, 45, 2);
    rendererRef.current = new CanvasRenderer();
    dispatcherRef.current = new CropDispatcher({ minRecognitionSize: 120 });

    return () => {
      stopCamera();
    };
  }, []);

  // Mode Change Trigger: Reset all active tracking identities & popups to evaluate fresh for new mode
  useEffect(() => {
    if (trackerRef.current) {
      trackerRef.current.clear();
    }
    activePopupRef.current = null;
  }, [systemMode]);

  const handleMatch = useCallback((eventData) => {
    activePopupRef.current = eventData;
    if (onAttendanceEvent && eventData.event_recorded) {
      onAttendanceEvent();
    }
  }, [onAttendanceEvent]);

  const frameDecimationCounterRef = useRef(0);

  // Main Live Loop with 1/3 Temporal Decimation (1 Real Keyframe + 2 Synthesized Fake Motion Frames)
  const runLiveLoop = useCallback(() => {
    if (!videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d', { alpha: false });
    const tFrameStart = performance.now();

    if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && video.videoWidth > 0) {
      const vw = video.videoWidth;
      const vh = video.videoHeight;

      if (canvas.width !== vw || canvas.height !== vh) {
        canvas.width = vw;
        canvas.height = vh;
      }

      // 1. Draw camera video frame to canvas
      ctx.drawImage(video, 0, 0, vw, vh);

      // 2. Measure Live Client FPS
      const now = performance.now();
      telemetryRef.current.frameCount++;
      if (now - telemetryRef.current.lastFpsTime >= 1000) {
        telemetryRef.current.fps = (telemetryRef.current.frameCount * 1000) / (now - telemetryRef.current.lastFpsTime);
        telemetryRef.current.frameCount = 0;
        telemetryRef.current.lastFpsTime = now;
      }

      // 3. 1/3rd Frame Decimation Logic:
      // Frame 0: REAL Detection Keyframe (Run ONNX + Dispatch Real Video Frame to Cloud GPU)
      // Frame 1 & 2: FAKE Motion Extrapolation (Skip ONNX + Skip Cloud Network Dispatches to save CPU/Battery)
      frameDecimationCounterRef.current = (frameDecimationCounterRef.current + 1) % 3;
      const isRealKeyframe = (frameDecimationCounterRef.current === 0);

      const detector = detectorRef.current;
      const tracker = trackerRef.current;

      if (isRealKeyframe) {
        // === REAL KEYFRAME (1/3rd) ===
        if (detector && detector.isReady && !detector.isBusy) {
          detector.isBusy = true;
          const t0 = performance.now();
          detector.detect(video, vw, vh).then((dets) => {
            detector.isBusy = false;
            telemetryRef.current.detectMs = performance.now() - t0;
            if (dets && tracker) {
              const tTrackS = performance.now();
              tracker.update(dets);
              telemetryRef.current.trackMs = performance.now() - tTrackS;
            }
          }).catch(() => { detector.isBusy = false; });
        }

        // Dispatch face crop extracted STRICTLY from the REAL video frame
        if (dispatcherRef.current && tracker) {
          dispatcherRef.current.checkAndDispatch(
            video,
            canvas,
            tracker,
            vw,
            vh,
            now,
            systemMode,
            handleMatch,
            handleMatch
          );
        }
      } else {
        // === FAKE EXTRAPOLATION FRAMES (2/3rd) ===
        // Zero ONNX compute overhead; smoothly glide bounding boxes via velocity dead-reckoning
        if (tracker) {
          tracker.extrapolateMotion();
        }
        // NOTE: dispatcher.checkAndDispatch is intentionally skipped here so NO fake frame is ever sent to server!
      }

      // 4. Render 60 FPS Tracking Overlay & HUDs
      if (rendererRef.current && tracker) {
        rendererRef.current.render(
          ctx,
          vw,
          vh,
          tracker.tracks,
          systemMode,
          telemetryRef.current,
          activePopupRef.current,
          now
        );
      }

      telemetryRef.current.e2eMs = performance.now() - tFrameStart;
    }

    animFrameRef.current = requestAnimationFrame(runLiveLoop);
  }, [systemMode, handleMatch]);

  const startCamera = async () => {
    setCameraError(null);
    setIsStartingEngine(true);

    if (detectorRef.current && !detectorRef.current.isReady && !detectorRef.current.isLoading) {
      try {
        await detectorRef.current.load();
      } catch (err) {
        console.warn('UltraLightDetector load notice:', err);
      }
    }

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setCameraError('Live camera requires HTTPS or modern WebRTC support.');
      setIsStartingEngine(false);
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: facingMode },
          width: { ideal: 640, max: 640 },
          height: { ideal: 480, max: 480 },
          frameRate: { ideal: 30, max: 30 }
        },
        audio: false
      });

      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.setAttribute('playsinline', 'true');
        await videoRef.current.play();
      }

      setIsCameraActive(true);
      setIsStartingEngine(false);
      animFrameRef.current = requestAnimationFrame(runLiveLoop);
    } catch (err) {
      setIsStartingEngine(false);
      setCameraError(`Failed to access camera: ${err.message}`);
    }
  };

  const stopCamera = () => {
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    if (trackerRef.current) {
      trackerRef.current.clear();
    }
    setIsCameraActive(false);
  };

  const toggleCamera = () => {
    if (isCameraActive) {
      stopCamera();
    } else {
      startCamera();
    }
  };

  const flipCamera = async () => {
    const newMode = facingMode === 'user' ? 'environment' : 'user';
    setFacingMode(newMode);
    stopCamera();
    setTimeout(() => {
      startCamera();
    }, 100);
  };

  return (
    <div className="video-viewport-card">
      <div className="viewport-header">
        <div className="viewport-status">
          <span className="pulse-indicator online" />
          <span className="viewport-label">
            LIVE CAMERA (ULTRA-LIGHT 1MB & IoU TRACKER)
          </span>
        </div>

        {isCameraActive && (
          <div className="camera-controls">
            <button className="ctrl-btn" onClick={flipCamera} title="Flip Camera">
              <RefreshCw size={14} />
              <span>Flip Cam</span>
            </button>
            <button className="ctrl-btn" onClick={toggleCamera} title="Pause / Stop">
              <Pause size={14} />
              <span>Pause</span>
            </button>
          </div>
        )}
      </div>

      <div className="viewport-container">
        {/* Hidden Video element for WebRTC camera stream */}
        <video ref={videoRef} className="hidden-video-elem" muted playsInline />

        <canvas ref={canvasRef} className="live-canvas-viewport" />

        {!isCameraActive && (
          <div className="camera-overlay">
            <div className="camera-overlay-content">
              <div className="overlay-icon-wrapper">
                <Camera size={38} />
              </div>
              <h3>Live Camera Recognition</h3>
              <p>
                Real-time face detection & tracking running locally with 120px recognition gate and instant cloud verification.
              </p>

              {cameraError && (
                <div className="camera-error-banner">
                  <AlertCircle size={16} />
                  <span>{cameraError}</span>
                </div>
              )}

              <button
                className="start-camera-btn"
                onClick={startCamera}
                disabled={isStartingEngine}
              >
                {isStartingEngine ? (
                  <>
                    <RefreshCw size={18} className="spin-icon" />
                    <span>Starting Ultra-Light Engine...</span>
                  </>
                ) : (
                  <>
                    <Camera size={18} />
                    <span>Start Live Capture</span>
                  </>
                )}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
