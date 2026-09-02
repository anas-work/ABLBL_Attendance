import os
import sys
import subprocess
import argparse

def build_tensorrt_engine(onnx_path: str, engine_path: str, precision: str = "fp16", max_batch_size: int = 1):
    """
    Compiles ONNX model into hardware-specific TensorRT FP16 / INT8 engine using trtexec or python bindings.
    Supports both NVIDIA Server GPUs (RTX A4000) and NVIDIA Jetson edge devices.
    """
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"Source ONNX file does not exist: {onnx_path}")

    print(f"Building TensorRT Engine: {onnx_path} -> {engine_path} (Precision: {precision.upper()})...")

    # Try trtexec command-line tool first (installed with TensorRT / JetPack SDK)
    trtexec_cmd = ["trtexec", f"--onnx={onnx_path}", f"--saveEngine={engine_path}"]
    if precision.lower() == "fp16":
        trtexec_cmd.append("--fp16")
    elif precision.lower() == "int8":
        trtexec_cmd.append("--int8")

    try:
        res = subprocess.run(trtexec_cmd, capture_output=True, text=True)
        if res.returncode == 0:
            print(f"  [SUCCESS] TensorRT Engine compiled at {engine_path} via trtexec.")
            return True
        else:
            print(f"  [INFO] trtexec output: {res.stderr[:200]}")
    except Exception as e:
        print(f"  [INFO] trtexec not found or failed ({e}). Attempting Python TensorRT API...")

    # Fallback to Python tensorrt API
    try:
        import tensorrt as trt
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        config = builder.create_builder_config()
        parser = trt.OnnxParser(network, TRT_LOGGER)

        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    print(f"  [ERROR] ONNX parse error: {parser.get_error(error)}")
                return False

        if precision.lower() == "fp16" and builder.platform_has_tf32:
            config.set_flag(trt.BuilderFlag.FP16)

        plan = builder.build_serialized_network(network, config)
        if plan is not None:
            with open(engine_path, 'wb') as f:
                f.write(plan)
            print(f"  [SUCCESS] TensorRT Engine built via Python API at {engine_path}")
            return True
    except Exception as e:
        print(f"  [NOTE] Python TensorRT API build note: {e}")
        print(f"  [READY FOR JETSON] On Jetson edge devices, execute: trtexec --onnx={onnx_path} --saveEngine={engine_path} --fp16")
        return False

def main():
    parser = argparse.ArgumentParser(description="Compile ONNX models to TensorRT FP16 engines.")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp16", "fp32", "int8"])
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)
    build_tensorrt_engine("models/scrfd_2.5g_kps.onnx", "models/scrfd_2.5g_kps.engine", precision=args.precision)
    build_tensorrt_engine("models/kprpe_adaface.onnx", "models/kprpe_adaface.engine", precision=args.precision)

if __name__ == "__main__":
    main()
