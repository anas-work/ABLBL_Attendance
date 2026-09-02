/**
 * High-Performance Client IoU Tracker with Motion Velocity Prediction & Permanent Identity Locking
 */

export class ClientIoUTracker {
  constructor(iouThreshold = 0.12, maxAge = 45, minHits = 2) {
    this.iouThreshold = iouThreshold;
    this.maxAge = maxAge;
    this.minHits = minHits;
    this.nextId = 101;
    this.tracks = [];
    this.frameCount = 0;
  }

  extrapolateMotion() {
    this.frameCount++;
    for (const t of this.tracks) {
      t.age++;
      t.time_since_update++;
      // Smooth dead-reckoning motion interpolation only when moving distinctly
      const speed = Math.hypot(t.vx || 0, t.vy || 0);
      if (t.time_since_update > 0 && t.time_since_update <= 8 && speed >= 1.5) {
        const dx = (t.vx || 0) * 0.5;
        const dy = (t.vy || 0) * 0.5;
        t.bbox[0] += dx;
        t.bbox[1] += dy;
        t.bbox[2] += dx;
        t.bbox[3] += dy;
      }
      // Velocity decay
      t.vx = (t.vx || 0) * 0.6;
      t.vy = (t.vy || 0) * 0.6;
    }
    return this.tracks.filter(t => t.time_since_update <= 15);
  }

  update(detections) {
    this.frameCount++;

    for (const t of this.tracks) {
      t.age++;
      t.time_since_update++;
    }

    if (detections.length === 0) {
      this.tracks = this.tracks.filter(t => t.time_since_update <= this.maxAge);
      return this.tracks.filter(t => t.time_since_update <= 15);
    }

    if (this.tracks.length === 0) {
      const matchedTracks = [];
      for (const d of detections) {
        const newT = {
          track_id: this.nextId++,
          bbox: [...d.bbox],
          score: d.score,
          kps: d.kps || [],
          hits: 1,
          age: 1,
          time_since_update: 0,
          vx: 0,
          vy: 0,
          assigned_identity: null,
          employee_id: null,
          similarity_score: 0.0,
          confirmed: false,
          recognition_stale: true,
          last_recognition_frame: 0,
          recognition_state: 'UNKNOWN',
          photo_url: null,
          decision: 'UNKNOWN',
          cropInFlight: false,
          lastCropAttemptTime: 0
        };
        this.tracks.push(newT);
        matchedTracks.push(newT);
      }
      return matchedTracks;
    }

    // Compute Generalized Association Matrix (IoU + Proximity)
    const iouMatrix = [];
    for (let i = 0; i < this.tracks.length; i++) {
      iouMatrix[i] = [];
      for (let j = 0; j < detections.length; j++) {
        const iou = this._computeIoU(this.tracks[i].bbox, detections[j].bbox);
        const tCx = (this.tracks[i].bbox[0] + this.tracks[i].bbox[2]) / 2.0;
        const tCy = (this.tracks[i].bbox[1] + this.tracks[i].bbox[3]) / 2.0;
        const dCx = (detections[j].bbox[0] + detections[j].bbox[2]) / 2.0;
        const dCy = (detections[j].bbox[1] + detections[j].bbox[3]) / 2.0;
        const dist = Math.hypot(tCx - dCx, tCy - dCy);
        const tSize = Math.max(this.tracks[i].bbox[2] - this.tracks[i].bbox[0], this.tracks[i].bbox[3] - this.tracks[i].bbox[1]);
        const dSize = Math.max(detections[j].bbox[2] - detections[j].bbox[0], detections[j].bbox[3] - detections[j].bbox[1]);
        const avgSize = Math.max(30, (tSize + dSize) / 2.0);

        const proxScore = Math.max(0, 1.0 - (dist / (avgSize * 1.5)));
        iouMatrix[i][j] = iou * 0.65 + proxScore * 0.35;
      }
    }

    const matchedTrackIndices = new Set();
    const matchedDetIndices = new Set();

    while (true) {
      let maxIoU = -1;
      let maxT = -1;
      let maxD = -1;

      for (let t = 0; t < this.tracks.length; t++) {
        if (matchedTrackIndices.has(t)) continue;
        for (let d = 0; d < detections.length; d++) {
          if (matchedDetIndices.has(d)) continue;
          if (iouMatrix[t][d] > maxIoU) {
            maxIoU = iouMatrix[t][d];
            maxT = t;
            maxD = d;
          }
        }
      }

      if (maxIoU < this.iouThreshold || maxT === -1) break;

      matchedTrackIndices.add(maxT);
      matchedDetIndices.add(maxD);

      const track = this.tracks[maxT];
      const det = detections[maxD];

      const oldW = track.bbox[2] - track.bbox[0];
      const oldH = track.bbox[3] - track.bbox[1];
      const newW = det.bbox[2] - det.bbox[0];
      const newH = det.bbox[3] - det.bbox[1];

      const oldCx = (track.bbox[0] + track.bbox[2]) / 2.0;
      const oldCy = (track.bbox[1] + track.bbox[3]) / 2.0;
      const newCx = (det.bbox[0] + det.bbox[2]) / 2.0;
      const newCy = (det.bbox[1] + det.bbox[3]) / 2.0;

      const centerShift = Math.hypot(newCx - oldCx, newCy - oldCy);

      // Deadzone filter: Ignore tiny sub-3.5px frame-to-frame sensor jitter
      let smoothCx, smoothCy;
      if (centerShift < 3.5) {
        smoothCx = oldCx;
        smoothCy = oldCy;
        track.vx = (track.vx || 0) * 0.5;
        track.vy = (track.vy || 0) * 0.5;
      } else {
        smoothCx = 0.35 * newCx + 0.65 * oldCx;
        smoothCy = 0.35 * newCy + 0.65 * oldCy;
        const instVx = (newCx - oldCx) * 0.30;
        const instVy = (newCy - oldCy) * 0.30;
        track.vx = (track.vx || 0) * 0.6 + instVx * 0.4;
        track.vy = (track.vy || 0) * 0.6 + instVy * 0.4;
      }

      // Heavy dimension stabilization (80% previous size, 20% new detection)
      const smoothW = 0.20 * newW + 0.80 * oldW;
      const smoothH = 0.20 * newH + 0.80 * oldH;

      track.bbox = [
        smoothCx - smoothW / 2.0,
        smoothCy - smoothH / 2.0,
        smoothCx + smoothW / 2.0,
        smoothCy + smoothH / 2.0
      ];
      track.score = det.score;
      if (det.kps && det.kps.length > 0) track.kps = det.kps;
      track.hits++;
      track.time_since_update = 0;
    }

    for (let d = 0; d < detections.length; d++) {
      if (!matchedDetIndices.has(d)) {
        const det = detections[d];
        const newT = {
          track_id: this.nextId++,
          bbox: [...det.bbox],
          score: det.score,
          kps: det.kps || [],
          hits: 1,
          age: 1,
          time_since_update: 0,
          vx: 0,
          vy: 0,
          assigned_identity: null,
          employee_id: null,
          similarity_score: 0.0,
          confirmed: false,
          recognition_stale: true,
          last_recognition_frame: 0,
          recognition_state: 'UNKNOWN',
          photo_url: null,
          decision: 'UNKNOWN',
          cropInFlight: false,
          lastCropAttemptTime: 0
        };
        this.tracks.push(newT);
      }
    }

    // Inter-track containment & proximity deduplication
    const validTracks = [];
    for (let i = 0; i < this.tracks.length; i++) {
      const tA = this.tracks[i];
      if (tA.time_since_update > this.maxAge) continue;
      let isDuplicate = false;
      for (let j = 0; j < validTracks.length; j++) {
        const tB = validTracks[j];
        const iou = this._computeIoU(tA.bbox, tB.bbox);
        const iom = this._computeIoM(tA.bbox, tB.bbox);
        const cAx = (tA.bbox[0] + tA.bbox[2]) / 2.0;
        const cAy = (tA.bbox[1] + tA.bbox[3]) / 2.0;
        const cBx = (tB.bbox[0] + tB.bbox[2]) / 2.0;
        const cBy = (tB.bbox[1] + tB.bbox[3]) / 2.0;
        const centerDist = Math.hypot(cAx - cBx, cAy - cBy);
        const avgSize = ((tA.bbox[2] - tA.bbox[0]) + (tB.bbox[2] - tB.bbox[0])) / 2.0;

        if (iou >= 0.15 || iom >= 0.25 || centerDist < avgSize * 0.75) {
          if (tA.recognition_state === 'MATCHED' && tB.recognition_state !== 'MATCHED') {
            tB.recognition_state = 'MATCHED';
            tB.assigned_identity = tA.assigned_identity;
            tB.employee_id = tA.employee_id;
            tB.similarity_score = tA.similarity_score;
            tB.photo_url = tA.photo_url;
            tB.decision = tA.decision;
            tB.confirmed = true;
          }
          isDuplicate = true;
          break;
        }
      }
      if (!isDuplicate) validTracks.push(tA);
    }
    this.tracks = validTracks;
    return this.tracks.filter(t => t.time_since_update <= 15);
  }

  _computeIoU(boxA, boxB) {
    const xA = Math.max(boxA[0], boxB[0]);
    const yA = Math.max(boxA[1], boxB[1]);
    const xB = Math.min(boxA[2], boxB[2]);
    const yB = Math.min(boxA[3], boxB[3]);

    const interArea = Math.max(0, xB - xA + 1) * Math.max(0, yB - yA + 1);
    const boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1);
    const boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1);

    return interArea / (boxAArea + boxBArea - interArea);
  }

  _computeIoM(boxA, boxB) {
    const xA = Math.max(boxA[0], boxB[0]);
    const yA = Math.max(boxA[1], boxB[1]);
    const xB = Math.min(boxA[2], boxB[2]);
    const yB = Math.min(boxA[3], boxB[3]);
    const inter = Math.max(0, xB - xA) * Math.max(0, yB - yA);
    const areaA = Math.max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]));
    const areaB = Math.max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]));
    return inter / Math.min(areaA, areaB);
  }

  clear() {
    this.tracks = [];
  }
}
