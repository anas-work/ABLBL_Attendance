import numpy as np
import onnx2tf.utils.common_functions
import onnx2tf
import sys

# Monkey-patch download_test_image_data to return dummy data
def patched_download_test_image_data():
    return np.zeros((1, 240, 320, 3), dtype=np.float32)

onnx2tf.utils.common_functions.download_test_image_data = patched_download_test_image_data
import onnx2tf.onnx2tf
onnx2tf.onnx2tf.download_test_image_data = patched_download_test_image_data

# Set arguments for onnx2tf
sys.argv = [
    'onnx2tf',
    '-i', 'app/src/main/assets/version-RFB-320.onnx',
    '-o', 'app/src/main/assets/version-RFB-320_tflite',
    '-nuo',
    '-kt', 'input' # force NHWC
]

onnx2tf.main()
