import subprocess
import os

input_onnx = "app/src/main/assets/version-RFB-320.onnx"
output_dir = "tflite_output"

if not os.path.exists(input_onnx):
    print(f"Error: {input_onnx} not found")
    exit(1)

print("Starting conversion with onnx2tf...")
try:
    # onnx2tf -i input.onnx -o output_dir
    subprocess.run(["onnx2tf", "-i", input_onnx, "-o", output_dir, "-non_verbose"], check=True)

    # Locate the .tflite file in the output directory
    tflite_path = os.path.join(output_dir, "version-RFB-320_float32.tflite")
    if os.path.exists(tflite_path):
        target_path = "app/src/main/assets/version-RFB-320.tflite"
        os.rename(tflite_path, target_path)
        print(f"Success! Model saved to {target_path}")
    else:
        print("Error: TFLite file not found in output directory")
except Exception as e:
    print(f"Conversion failed: {e}")
