import os
import cv2
import numpy as np
import onnxruntime as ort
from typing import Optional

class KPRPEAdaFaceRecognizer:
    """
    KPRPE + AdaFace Feature Embedding Extraction Engine.
    Takes 112x112 aligned facial crops and generates L2-normalized 512-d embeddings.
    """

    def __init__(
        self,
        model_path: str = "models/kprpe_adaface.onnx",
        embedding_size: int = 512,
        device: str = "cuda"
    ):
        self.model_path = model_path
        self.embedding_size = embedding_size

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"KPRPE AdaFace model ONNX file not found at: {model_path}")

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == "cuda" else ['CPUExecutionProvider']
        
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        try:
            self.session = ort.InferenceSession(model_path, opts, providers=providers)
        except Exception:
            self.session = ort.InferenceSession(model_path, opts, providers=['CPUExecutionProvider'])

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def extract_embedding(self, aligned_face_112: np.ndarray) -> np.ndarray:
        """
        Extracts 512-d L2-normalized facial embedding from a 112x112 aligned RGB or BGR image.
        """
        if aligned_face_112.shape[:2] != (112, 112):
            aligned_face_112 = cv2.resize(aligned_face_112, (112, 112))

        # Convert to RGB if needed and normalize: (img - 127.5) / 128.0
        if aligned_face_112.ndim == 3 and aligned_face_112.shape[2] == 3:
            # Assuming BGR input from OpenCV, convert to RGB
            rgb_img = cv2.cvtColor(aligned_face_112, cv2.COLOR_BGR2RGB)
        else:
            rgb_img = aligned_face_112

        blob = ((rgb_img.astype(np.float32) - 127.5) / 128.0).transpose(2, 0, 1)
        blob = np.expand_dims(blob, axis=0)  # Shape: (1, 3, 112, 112)

        out = self.session.run([self.output_name], {self.input_name: blob})[0]
        embedding = out.flatten()

        # L2 Normalization
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding
