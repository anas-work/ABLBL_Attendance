/**
 * Crop Dispatcher Engine
 * Gating & Sampling Rules:
 * 1. 120px Gate: Activates only when face size >= 120px.
 * 2. 1-Second Settle-Down Delay: Holds steady for 1000ms to allow user focus & stabilization before sampling.
 * 3. 5 Evenly-Spaced Frames: Dispatches at most 5 frames spaced by ~600ms over a 3.0s window.
 * 4. Final Unknown Classification: Flags as UNKNOWN only after all 5 attempts fail and total elapsed window >= 4.0s.
 */

export class CropDispatcher {
  constructor(options = {}) {
    this.minRecognitionSize = options.minRecognitionSize || 120;
    this.settleDelayMs = options.settleDelayMs || 1000; // 1 second settling time
    this.samplingIntervalMs = options.samplingIntervalMs || 600; // 5 frames over 3000ms = 600ms interval
    this.maxAttempts = 5;
    this.totalEvaluationWindowMs = 4000; // 1s settle + 3s sampling
    this.cropWidth = 224;
    this.cropHeight = 224;

    this.cropCanvas = document.createElement('canvas');
    this.cropCanvas.width = this.cropWidth;
    this.cropCanvas.height = this.cropHeight;
    this.cropCtx = this.cropCanvas.getContext('2d', { willReadFrequently: true, alpha: false });
  }

  async checkAndDispatch(videoElem, liveCanvasElem, tracker, vw, vh, now, currentSystemMode, onMatch, onUnknown) {
    const candidateTracks = tracker.tracks
      .filter(t => t.time_since_update <= 8)
      .sort((a, b) => {
        const areaA = (a.bbox[2] - a.bbox[0]) * (a.bbox[3] - a.bbox[1]);
        const areaB = (b.bbox[2] - b.bbox[0]) * (b.bbox[3] - b.bbox[1]);
        return areaB - areaA;
      });

    for (let i = 0; i < candidateTracks.length; i++) {
      const track = candidateTracks[i];
      const boxW = track.bbox[2] - track.bbox[0];
      const boxH = track.bbox[3] - track.bbox[1];
      const faceSize = Math.max(boxW, boxH);

      // If already matched with high certainty, lock and do not resend
      if (track.recognition_state === 'MATCHED') continue;

      if (faceSize < this.minRecognitionSize) {
        track.firstSeenAt120px = null; // Reset evaluation window if person moves far away
        if (track.recognition_state !== 'NOT_RECOGNIZED') {
          track.recognition_state = 'WAITING_FOR_SIZE';
        }
        continue;
      }

      // Ignore cut-off faces entering from outer edge of screen
      if (track.bbox[0] <= 6 || track.bbox[1] <= 6 || track.bbox[2] >= vw - 6 || track.bbox[3] >= vh - 6) {
        continue;
      }

      // 1. Initial Gate Trigger: Mark the timestamp when person reaches >=120px
      if (!track.firstSeenAt120px) {
        track.firstSeenAt120px = now;
        track.evalAttempts = 0;
        track.lastCropAttemptTime = 0;
        track.recognition_state = 'SETTLING';
      }

      const elapsedSinceGate = now - track.firstSeenAt120px;

      // 2. Settle-Down Delay: Wait for 1 full second to let the person settle down
      if (elapsedSinceGate < this.settleDelayMs) {
        track.recognition_state = 'SETTLING';
        continue;
      }

      // Prioritize top 2 prominent faces
      if (i >= 2) continue;

      // 3. Controlled Sampling: Send at most 5 frames spaced by ~600ms within the 3-second evaluation window
      const canSample = !track.cropInFlight &&
        (track.evalAttempts < this.maxAttempts) &&
        (now - (track.lastCropAttemptTime || 0) >= this.samplingIntervalMs);

      if (canSample) {
        track.cropInFlight = true;
        track.lastCropAttemptTime = now;
        track.evalAttempts = (track.evalAttempts || 0) + 1;
        track.recognition_state = 'RECOGNIZING';

        // Extract clean 25% padded crop from the REAL video frame
        const padX = Math.max(10, Math.round(boxW * 0.25));
        const padY = Math.max(10, Math.round(boxH * 0.25));
        const cropX = Math.max(0, Math.round(track.bbox[0] - padX));
        const cropY = Math.max(0, Math.round(track.bbox[1] - padY));
        const cropW = Math.max(30, Math.min(vw - cropX, Math.round(boxW + padX * 2)));
        const cropH = Math.max(30, Math.min(vh - cropY, Math.round(boxH + padY * 2)));

        this.cropCtx.drawImage(videoElem, cropX, cropY, cropW, cropH, 0, 0, this.cropWidth, this.cropHeight);
        const cropBase64 = this.cropCanvas.toDataURL('image/jpeg', 0.92);

        this._sendCrop(track, cropBase64, currentSystemMode, onMatch, onUnknown, now);
      }
    }
  }

  async _sendCrop(track, cropBase64, currentSystemMode, onMatch, onUnknown, dispatchTime) {
    try {
      const resp = await fetch('/api/process_crop', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          crop_base64: cropBase64,
          full_frame_base64: cropBase64
        })
      });

      if (resp.ok) {
        const data = await resp.json();

        if (data.matched) {
          track.recognition_state = 'MATCHED';
          track.assigned_identity = data.name;
          track.employee_id = data.employee_id;
          track.similarity_score = Number(data.confidence) || 0.85;
          track.decision = data.decision || 'CHECK_IN';
          track.photo_url = data.photo_url || null;

          if (onMatch) {
            onMatch({
              track_id: track.track_id,
              name: data.name,
              employee_id: data.employee_id,
              similarity: Number(data.confidence) || 0.85,
              confidence: Number(data.confidence) || 0.85,
              decision: data.decision || 'CHECK_IN',
              photo_url: data.photo_url || null,
              until: performance.now() + 4000,
              event_recorded: data.event_recorded || false
            });
          }
        } else {
          // Check if all 5 attempts have been exhausted and total evaluation time has passed
          const totalEvalTime = performance.now() - (track.firstSeenAt120px || performance.now());
          const attempts = track.evalAttempts || 0;

          if (attempts >= this.maxAttempts && totalEvalTime >= this.totalEvaluationWindowMs) {
            if (track.recognition_state !== 'NOT_RECOGNIZED') {
              track.recognition_state = 'NOT_RECOGNIZED';
              track.assigned_identity = 'UNKNOWN PERSON';
              track.employee_id = 'UNKNOWN';
              track.similarity_score = 0.0;

              // Record unknown incident to backend
              fetch('/api/record_unknown', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                  crop_base64: cropBase64,
                  full_frame_base64: cropBase64
                })
              }).then((r) => r.json()).then((unkData) => {
                if (onUnknown) {
                  onUnknown({
                    track_id: track.track_id,
                    name: 'UNKNOWN PERSON',
                    employee_id: 'UNKNOWN',
                    similarity: 0.0,
                    confidence: 0.0,
                    decision: 'UNKNOWN',
                    is_unknown: true,
                    photo_url: unkData.crop_image_path ? `/photos/${unkData.crop_image_path.split('/').pop()}` : null,
                    until: performance.now() + 4000,
                    event_recorded: true
                  });
                }
              }).catch((e) => console.warn('Record unknown warning:', e));
            }
          }
        }
      }
    } catch (err) {
      console.warn('Process crop network error:', err);
    } finally {
      track.cropInFlight = false;
    }
  }
}
