import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import os

onnx_model_path = 'app/src/main/assets/version-RFB-320.onnx'
tf_model_path = 'app/src/main/assets/version-RFB-320_tf'
tflite_model_path = 'app/src/main/assets/version-RFB-320.tflite'

# Load the ONNX model
onnx_model = onnx.load(onnx_model_path)

# Prepare the TensorFlow backend
tf_rep = prepare(onnx_model)

# Export the model to TensorFlow SavedModel format
tf_rep.export_graph(tf_model_path)

# Convert the SavedModel to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
tflite_model = converter.convert()

# Save the TFLite model
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f"Successfully converted {onnx_model_path} to {tflite_model_path}")
