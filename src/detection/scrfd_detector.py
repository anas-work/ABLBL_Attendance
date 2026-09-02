import os
import cv2
import numpy as np
import onnxruntime as ort
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class FaceDetection:
    bbox: np.ndarray      # [x1, y1, x2, y2]
    score: float          # confidence score
    kps: np.ndarray       # 5x2 array of landmark keypoints [[x, y], ...]

class SCRFDDetector:
    """
    SCRFD Face Detector & 5-Landmark Extractor.
    High performance face detection optimized for RTX A4000 and Jetson TensorRT/ONNX.
    """

    def __init__(
        self,
        model_path: str = "models/scrfd_2.5g_kps.onnx",
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        input_size: Tuple[int, int] = (640, 640),
        device: str = "cuda"
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"SCRFD model ONNX file not found at: {model_path}")

        # Choose execution providers
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == "cuda" else ['CPUExecutionProvider']
        
        # Configure ONNX Runtime session
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        try:
            self.session = ort.InferenceSession(model_path, opts, providers=providers)
        except Exception:
            # Fallback to CPU execution if CUDA provider initialization has library conflicts
            self.session = ort.InferenceSession(model_path, opts, providers=['CPUExecutionProvider'])

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [out.name for out in self.session.get_outputs()]

        self.fmc = 3
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        self.use_kps = True

        # Pre-compute and cache anchor centers for each stride (constant given fixed input_size)
        det_w, det_h = self.input_size  # unpack here so the cache loop can reference them
        self._anchor_centers_cache: dict = {}
        for stride in self._feat_stride_fpn:
            height = det_h // stride
            width = det_w // stride
            ac = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
            ac = (ac * stride).reshape((-1, 2))
            if self._num_anchors > 1:
                ac = np.stack([ac] * self._num_anchors, axis=1).reshape((-1, 2))
            self._anchor_centers_cache[stride] = ac

    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        """
        Detects faces and 5 facial keypoints in an input image (BGR uint8).
        Returns a list of FaceDetection objects.
        """
        img_h, img_w, _ = image.shape
        det_w, det_h = self.input_size

        # Preprocessing: resize with aspect ratio letterboxing
        r = min(det_w / img_w, det_h / img_h)
        new_w, new_h = int(img_w * r), int(img_h * r)
        resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        det_img = np.zeros((det_h, det_w, 3), dtype=np.uint8)
        det_img[0:new_h, 0:new_w] = resized_img

        # Normalize RGB: (X - 127.5) / 128.0
        blob = cv2.dnn.blobFromImage(det_img, 1.0 / 128.0, (det_w, det_h), (127.5, 127.5, 127.5), swapRB=True)

        net_outs = self.session.run(self.output_names, {self.input_name: blob})

        scores_list = []
        bboxes_list = []
        kpss_list = []

        num_strides = len(self._feat_stride_fpn)
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = net_outs[idx]
            bbox_preds = net_outs[idx + num_strides]
            bbox_preds = bbox_preds * stride

            if self.use_kps:
                kps_preds = net_outs[idx + num_strides * 2] * stride

            # Use pre-cached anchor centers (no recomputation needed)
            anchor_centers = self._anchor_centers_cache[stride]

            pos_inds = np.where(scores >= self.conf_threshold)[0]
            if len(pos_inds) == 0:
                continue

            scores = scores[pos_inds]
            bbox_preds = bbox_preds[pos_inds]

            # Decode bounding boxes
            x1 = anchor_centers[pos_inds, 0] - bbox_preds[:, 0]
            y1 = anchor_centers[pos_inds, 1] - bbox_preds[:, 1]
            x2 = anchor_centers[pos_inds, 0] + bbox_preds[:, 2]
            y2 = anchor_centers[pos_inds, 1] + bbox_preds[:, 3]
            
            bboxes = np.stack([x1, y1, x2, y2], axis=-1)

            if self.use_kps:
                kps_preds = kps_preds[pos_inds]
                kpss = np.zeros((kps_preds.shape[0], 5, 2), dtype=np.float32)
                for k in range(5):
                    kpss[:, k, 0] = anchor_centers[pos_inds, 0] + kps_preds[:, k * 2]
                    kpss[:, k, 1] = anchor_centers[pos_inds, 1] + kps_preds[:, k * 2 + 1]

            # Rescale back to original image coordinates
            bboxes = bboxes / r
            if self.use_kps:
                kpss = kpss / r

            scores_list.append(scores)
            bboxes_list.append(bboxes)
            if self.use_kps:
                kpss_list.append(kpss)

        if not scores_list:
            return []

        scores = np.vstack(scores_list)
        bboxes = np.vstack(bboxes_list)
        kpss = np.vstack(kpss_list) if self.use_kps else None

        # NMS filtering
        keep = self._nms(bboxes, scores.flatten(), self.nms_threshold)

        results = []
        for i in keep:
            results.append(FaceDetection(
                bbox=bboxes[i],
                score=float(scores[i]),
                kps=kpss[i] if kpss is not None else None
            ))

        return results

    @staticmethod
    def _nms(dets: np.ndarray, scores: np.ndarray, thresh: float) -> List[int]:
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep
