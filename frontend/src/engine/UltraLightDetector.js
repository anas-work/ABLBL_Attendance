/**
 * Linzaer Ultra-Light 1MB Face Detector (version-RFB-320) Engine
 * Highly optimized for low-end mobile CPUs and high-end devices:
 * - Pre-allocated reusable Float32Array buffers (Zero GC thrashing)
 * - Optimal WASM SIMD / WebGL execution provider selection
 * - Non-blocking asynchronous inference
 */

export class UltraLightDetector {
  constructor(options = {}) {
    this.modelPath = options.modelPath || '/models/ultra_light/version-RFB-320.onnx';
    this.confThreshold = options.confThreshold || 0.58;
    this.nmsThreshold = options.nmsThreshold || 0.25;
    this.width = 320;
    this.height = 240;
    this.session = null;
    this.priors = null;
    this.isReady = false;
    this.isLoading = false;
    this.isBusy = false;

    // Dedicated offscreen canvas for preprocessing
    this.canvas = document.createElement('canvas');
    this.canvas.width = this.width;
    this.canvas.height = this.height;
    this.ctx = this.canvas.getContext('2d', { willReadFrequently: true, alpha: false });

    // Pre-allocate static float buffer once to eliminate garbage collection pauses on mobile
    this.planeSize = this.width * this.height;
    this.floatData = new Float32Array(1 * 3 * this.height * this.width);
  }

  initPriors() {
    const min_boxes = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]];
    const steps = [8, 16, 32, 64];
    const priors = [];

    for (let k = 0; k < steps.length; k++) {
      const f_h = Math.ceil(this.height / steps[k]);
      const f_w = Math.ceil(this.width / steps[k]);
      const min_sizes = min_boxes[k];

      for (let i = 0; i < f_h; i++) {
        for (let j = 0; j < f_w; j++) {
          for (const min_size of min_sizes) {
            const s_kx = min_size / this.width;
            const s_ky = min_size / this.height;
            const cx = (j + 0.5) * steps[k] / this.width;
            const cy = (i + 0.5) * steps[k] / this.height;
            priors.push(cx, cy, s_kx, s_ky);
          }
        }
      }
    }
    this.priors = new Float32Array(priors);
  }

  async load() {
    if (this.isReady || this.isLoading) return;
    this.isLoading = true;
    this.initPriors();

    const ortInstance = typeof ort !== 'undefined' ? ort : (window.ort || null);
    if (!ortInstance) {
      this.isLoading = false;
      throw new Error('ONNX Runtime Web is not loaded.');
    }

    try {
      const cores = navigator.hardwareConcurrency || 2;
      ortInstance.env.wasm.numThreads = Math.min(2, Math.max(1, cores - 1));
      ortInstance.env.wasm.simd = true;

      // Try WebGL first for mobile GPU acceleration; fall back to WebAssembly
      this.session = await ortInstance.InferenceSession.create(this.modelPath, {
        executionProviders: ['webgl', 'wasm'],
        graphOptimizationLevel: 'all'
      });

      this.isReady = true;
      this.isLoading = false;
      console.log('Linzaer Ultra-Light 1MB Detector loaded with zero-copy memory optimization!');
    } catch (err) {
      // Fallback cleanly to pure WASM if WebGL is unavailable
      try {
        this.session = await ortInstance.InferenceSession.create(this.modelPath, {
          executionProviders: ['wasm'],
          graphOptimizationLevel: 'all'
        });
        this.isReady = true;
        this.isLoading = false;
        console.log('Linzaer Ultra-Light Detector loaded (WASM Fallback).');
      } catch (fallbackErr) {
        this.isLoading = false;
        throw fallbackErr;
      }
    }
  }

  async detect(videoElem, vw, vh) {
    if (!this.isReady || !this.session || !this.priors) return [];

    // Draw directly into fixed 320x240 offscreen canvas
    this.ctx.drawImage(videoElem, 0, 0, this.width, this.height);
    const imgData = this.ctx.getImageData(0, 0, this.width, this.height);
    const pixels = imgData.data;

    // Fast Planar Float32 NCHW Normalization into pre-allocated memory buffer:
    // (Pixel - 127.0) / 128.0 == Pixel * 0.0078125 - 0.9921875
    const floatData = this.floatData;
    const planeSize = this.planeSize;
    const plane2 = planeSize * 2;

    for (let i = 0; i < planeSize; i++) {
      const pxIdx = i << 2;
      floatData[i] = pixels[pxIdx] * 0.0078125 - 0.9921875;
      floatData[planeSize + i] = pixels[pxIdx + 1] * 0.0078125 - 0.9921875;
      floatData[plane2 + i] = pixels[pxIdx + 2] * 0.0078125 - 0.9921875;
    }

    const ortInstance = typeof ort !== 'undefined' ? ort : window.ort;
    const inputTensor = new ortInstance.Tensor('float32', floatData, [1, 3, this.height, this.width]);
    const netOuts = await this.session.run({ input: inputTensor });

    const scores = netOuts.scores.data;
    const boxes = netOuts.boxes.data;
    const numPriors = this.priors.length / 4;
    const detections = [];

    // Prior decoding & confidence filtering
    for (let i = 0; i < numPriors; i++) {
      const score = scores[i * 2 + 1];
      if (score >= this.confThreshold) {
        const priorIdx = i * 4;
        const boxIdx = i * 4;

        const pCx = this.priors[priorIdx];
        const pCy = this.priors[priorIdx + 1];
        const pW = this.priors[priorIdx + 2];
        const pH = this.priors[priorIdx + 3];

        const b0 = boxes[boxIdx];
        const b1 = boxes[boxIdx + 1];
        const b2 = boxes[boxIdx + 2];
        const b3 = boxes[boxIdx + 3];

        const cx = pCx + b0 * 0.1 * pW;
        const cy = pCy + b1 * 0.1 * pH;
        const w = pW * Math.exp(b2 * 0.2);
        const h = pH * Math.exp(b3 * 0.2);

        const x1 = Math.max(0, (cx - w / 2.0) * vw);
        const y1 = Math.max(0, (cy - h / 2.0) * vh);
        const x2 = Math.min(vw, (cx + w / 2.0) * vw);
        const y2 = Math.min(vh, (cy + h / 2.0) * vh);

        const rawW = x2 - x1;
        const rawH = y2 - y1;

        // User calibration: Increase height from top a little, reduce from bottom a little, expand width from both sides
        const padX = rawW * 0.08;
        let adjX1 = Math.max(0, x1 - padX);
        let adjX2 = Math.min(vw, x2 + padX);
        let adjY1 = Math.max(0, y1 - rawH * 0.10); // Expand from top
        let adjY2 = Math.min(vh, y2 - rawH * 0.08); // Reduce from bottom

        let finalW = adjX2 - adjX1;
        let finalH = adjY2 - adjY1;
        let finalAspect = finalW / Math.max(1, finalH);

        // Sanity gate: face must be at least 25x25 and valid aspect ratio
        if (finalW >= 25 && finalH >= 25 && (finalAspect >= 0.55 && finalAspect <= 1.55)) {
          detections.push({
            bbox: [adjX1, adjY1, adjX2, adjY2],
            score: score
          });
        }
      }
    }

    if (detections.length <= 1) {
      return detections;
    }

    // Fast greedy Non-Maximum Suppression (NMS)
    detections.sort((a, b) => b.score - a.score);
    const keep = [];
    const suppressed = new Uint8Array(detections.length);

    for (let i = 0; i < detections.length; i++) {
      if (suppressed[i]) continue;
      keep.push(detections[i]);

      const b1 = detections[i].bbox;
      const b1Area = (b1[2] - b1[0] + 1) * (b1[3] - b1[1] + 1);

      for (let j = i + 1; j < detections.length; j++) {
        if (suppressed[j]) continue;
        const b2 = detections[j].bbox;
        const xx1 = Math.max(b1[0], b2[0]);
        const yy1 = Math.max(b1[1], b2[1]);
        const xx2 = Math.min(b1[2], b2[2]);
        const yy2 = Math.min(b1[3], b2[3]);
        const w = Math.max(0, xx2 - xx1 + 1);
        const h = Math.max(0, yy2 - yy1 + 1);
        const inter = w * h;

        const b2Area = (b2[2] - b2[0] + 1) * (b2[3] - b2[1] + 1);
        const iou = inter / Math.max(1, b1Area + b2Area - inter);

        if (iou >= this.nmsThreshold) {
          suppressed[j] = 1;
        }
      }
    }

    return keep;
  }
}
