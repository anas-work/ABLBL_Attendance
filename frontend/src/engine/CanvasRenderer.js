/**
 * Canvas Overlay Renderer
 * Renders OpenCV-style bounding boxes, top-left telemetry HUD, and top-right official ID card popups.
 */

export class CanvasRenderer {
  constructor() {
    this.photoCache = {};
  }

  render(ctx, vw, vh, tracks, systemMode, telemetry, activeIdCardPopup, now) {
    if (!ctx) return;

    let activeTrackCount = 0;

    // 1. Draw Bounding Boxes
    for (const track of tracks) {
      if (track.time_since_update > 15) continue;
      activeTrackCount++;

      const bx = Math.round(track.bbox[0]);
      const by = Math.round(track.bbox[1]);
      const bw = Math.round(track.bbox[2] - track.bbox[0]);
      const bh = Math.round(track.bbox[3] - track.bbox[1]);

      let boxColor = '#38bdf8'; // Default Sky Blue
      let labelText = 'DETECTING...';

      if (track.recognition_state === 'MATCHED' && track.assigned_identity) {
        const scorePct = track.similarity_score > 0 ? ` [${Math.round(track.similarity_score * 100)}%]` : ' [VERIFIED]';
        const dec = track.decision;
        if (dec === 'CHECK_OUT') {
          boxColor = '#d946ef'; // Purple
          labelText = `CHECK-OUT: ${track.assigned_identity}${scorePct}`;
        } else if (dec === 'RE_ENTRY') {
          boxColor = '#f59e0b'; // Orange
          labelText = `RE-ENTRY: ${track.assigned_identity}${scorePct}`;
        } else {
          boxColor = '#10b981'; // Green
          labelText = `${track.assigned_identity}${scorePct}`;
        }
      } else if (track.recognition_state === 'NOT_RECOGNIZED') {
        boxColor = '#ef4444'; // Red Critical
        labelText = '⚠️ NOT RECOGNIZED';
      } else if (track.recognition_state === 'SETTLING') {
        boxColor = '#a855f7'; // Purple Settle
        labelText = 'HOLD STEADY...';
      } else if (track.recognition_state === 'WAITING_FOR_SIZE') {
        boxColor = '#64748b'; // Gray
        labelText = 'APPROACH CAMERA';
      } else if (track.recognition_state === 'RECOGNIZING') {
        boxColor = '#06b6d4'; // Cyan
        const attempt = track.evalAttempts ? ` (${track.evalAttempts}/5)` : '';
        labelText = `ANALYZING${attempt}...`;
      }

      // Draw bounding box (2.5px border)
      ctx.lineWidth = track.recognition_state === 'NOT_RECOGNIZED' ? 3 : 2;
      ctx.strokeStyle = boxColor;
      ctx.strokeRect(bx, by, bw, bh);

      // Draw label header banner
      ctx.font = 'bold 12px "DejaVu Sans", "Helvetica Neue", Arial, sans-serif';
      const textMetrics = ctx.measureText(labelText);
      const labelW = textMetrics.width + 12;
      const labelH = 24;
      const labelY = Math.max(0, by - labelH);

      ctx.fillStyle = boxColor;
      ctx.fillRect(bx, labelY, labelW, labelH);
      ctx.fillStyle = '#ffffff';
      ctx.fillText(labelText, bx + 6, labelY + 16);
    }

    // 2. Draw Top-Left Diagnostics HUD Box
    const hudW = 320;
    const hudH = 75;
    ctx.fillStyle = 'rgba(10, 15, 29, 0.88)';
    ctx.fillRect(10, 10, hudW, hudH);

    const hudBorderColor = systemMode === 'EXIT' ? '#d946ef' : '#10b981';
    ctx.strokeStyle = hudBorderColor;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(10, 10, hudW, hudH);

    ctx.font = '12px "DejaVu Sans", "Helvetica Neue", Arial, sans-serif';
    ctx.fillStyle = '#ffffff';
    ctx.fillText(`MODE: ${systemMode} | FPS: ${telemetry.fps.toFixed(1)} | Tracks: ${activeTrackCount}`, 18, 30);
    ctx.fillText(`Detect: ${telemetry.detectMs.toFixed(1)}ms | Track: ${telemetry.trackMs.toFixed(1)}ms`, 18, 50);
    ctx.fillText(`End-to-End Latency: ${telemetry.e2eMs.toFixed(1)}ms`, 18, 70);

    // 3. Draw Flashing Official ID Card HUD (Top Right)
    if (activeIdCardPopup && now < activeIdCardPopup.until) {
      const cardW = 360;
      const cardH = 130;
      const cardX = vw - cardW - 20;
      const cardY = 20;

      const isUnknown = activeIdCardPopup.is_unknown || activeIdCardPopup.decision === 'UNKNOWN';
      let cardBorderColor = '#10b981';
      let headerTitle = 'OFFICIAL ID CARD VERIFIED';
      let cardBg = 'rgba(15, 23, 42, 0.92)';

      if (isUnknown) {
        cardBorderColor = '#ef4444';
        headerTitle = '⚠️ CRITICAL: UNKNOWN PERSON DETECTED';
        cardBg = 'rgba(45, 10, 10, 0.95)';
      } else if (activeIdCardPopup.decision === 'CHECK_OUT') {
        cardBorderColor = '#d946ef';
        headerTitle = 'CHECK-OUT VERIFIED';
      } else if (activeIdCardPopup.decision === 'RE_ENTRY') {
        cardBorderColor = '#f59e0b';
        headerTitle = 'RE-ENTRY VERIFIED';
      }

      // Background Card
      ctx.fillStyle = cardBg;
      ctx.fillRect(cardX, cardY, cardW, cardH);
      ctx.strokeStyle = cardBorderColor;
      ctx.lineWidth = 3;
      ctx.strokeRect(cardX, cardY, cardW, cardH);

      // Photo Box (90x105)
      const photoX = cardX + 12;
      const photoY = cardY + 12;
      const photoW = 90;
      const photoH = 105;

      ctx.fillStyle = '#000000';
      ctx.fillRect(photoX, photoY, photoW, photoH);

      const photoUrl = activeIdCardPopup.photo_url;
      if (photoUrl) {
        if (!this.photoCache[photoUrl]) {
          const img = new Image();
          img.src = photoUrl;
          img.onload = () => { this.photoCache[photoUrl] = img; };
        }
        const cachedImg = this.photoCache[photoUrl];
        if (cachedImg && cachedImg.complete) {
          ctx.drawImage(cachedImg, photoX, photoY, photoW, photoH);
        } else {
          ctx.font = '28px sans-serif';
          ctx.fillStyle = '#64748b';
          ctx.fillText('👤', photoX + 30, photoY + 60);
        }
      } else {
        ctx.font = '28px sans-serif';
        ctx.fillStyle = isUnknown ? '#ef4444' : '#64748b';
        ctx.fillText(isUnknown ? '⚠️' : '👤', photoX + 30, photoY + 60);
      }

      ctx.strokeStyle = cardBorderColor;
      ctx.lineWidth = 1.5;
      ctx.strokeRect(photoX, photoY, photoW, photoH);

      // Text Info
      const textX = cardX + 115;
      ctx.font = 'bold 11px "DejaVu Sans", Arial, sans-serif';
      ctx.fillStyle = isUnknown ? '#f87171' : '#38bdf8';
      ctx.fillText(headerTitle, textX, cardY + 26);

      ctx.font = 'bold 13px "DejaVu Sans", Arial, sans-serif';
      ctx.fillStyle = '#ffffff';
      ctx.fillText(`IDENTITY: ${activeIdCardPopup.name || 'Unknown Person'}`, textX, cardY + 52);

      ctx.font = '12px "DejaVu Sans", Arial, sans-serif';
      ctx.fillStyle = '#cbd5e1';
      ctx.fillText(
        `STATUS: ${isUnknown ? 'UNREGISTERED / NOT RECOGNIZED' : activeIdCardPopup.decision || 'PRESENT'}`,
        textX,
        cardY + 74
      );

      // Fix 0% confidence bug by checking both similarity and confidence properly
      const rawScore = activeIdCardPopup.similarity ?? activeIdCardPopup.confidence ?? 0;
      const scorePct = rawScore > 0 ? `${Math.round(rawScore * 100)}%` : 'VERIFIED';

      ctx.font = '12px "JetBrains Mono", monospace';
      ctx.fillStyle = isUnknown ? '#ef4444' : '#34d399';
      ctx.fillText(
        isUnknown ? 'VERIFICATION: MATCH NOT FOUND' : `CONFIDENCE: ${scorePct}`,
        textX,
        cardY + 96
      );
    }
  }
}
