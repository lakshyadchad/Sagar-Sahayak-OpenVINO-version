import openvino as ov
from ultralytics import YOLO  # type: ignore
import os

# --- CONFIGURATION ---
# 1. Paths for Enhancer (U-Net)
ENHANCER_ONNX_PATH = ""
ENHANCER_OUTPUT_DIR = ""

# 2. Paths for Detector (YOLO)
# Point to the .pt file (It's safer to let Ultralytics handle the full conversion chain)
YOLO_PT_PATH = ""

def convert_enhancer():
    print(f"--- Converting Enhancer (U-Net) ---")
    
    # Initialize OpenVINO Core
    core = ov.Core()
    
    # 1. Read the ONNX model
    print(f"Reading {ENHANCER_ONNX_PATH}...")
    model = core.read_model(ENHANCER_ONNX_PATH)
    
    # 2. Convert & Save (FP16 Precision)
    # compress_to_fp16=True cuts model size in half with almost zero accuracy loss
    output_path = os.path.join(ENHANCER_OUTPUT_DIR, "enhancer_fp16.xml")
    os.makedirs(ENHANCER_OUTPUT_DIR, exist_ok=True)
    
    print("Saving to OpenVINO IR format (FP16)...")
    ov.save_model(model, output_path, compress_to_fp16=True)
    
    print(f"Success! Saved to: {output_path}")

def convert_detector():
    print(f"\n--- Converting Detector (YOLO) ---")
    
    # Ultralytics has a specialized exporter for OpenVINO
    # This handles the tricky "Resize" and "Concat" layers that usually break generic converters
    model = YOLO(YOLO_PT_PATH)
    
    print("Exporting YOLO to OpenVINO format...")
    # format='openvino' automatically enables FP16 (half=True is default)
    # dynamic=True allows the input size to change slightly if needed
    model.export(format='openvino', dynamic=True, half=True)
    
    print("Success! Ultralytics saved the model in the training folder.")

if __name__ == "__main__":
    convert_enhancer()
    convert_detector()