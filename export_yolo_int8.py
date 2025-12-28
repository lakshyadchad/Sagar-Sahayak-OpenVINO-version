from ultralytics import YOLO  # type: ignore

def export_int8():
    print("--- Starting YOLOv11 INT8 Export ---")

    # 1. Load your trained PyTorch model
    # Ensure this path points to your actual trained weights
    model_path = ""
    model = YOLO(model_path)

    # 2. Run the Export
    # format='openvino': Target format
    # int8=True: Enables Quantization (makes it 4x smaller)
    # data: CRITICAL. Tells NNCF where to find images for calibration.
    model.export(
        format='openvino', 
        int8=True, 
        data='',  # Absolute path to dataset config
        dynamic=True          # Good for OpenVINO input flexibility
    )

    print("Export Complete! Check the 'weights' folder for the '_int8' directory.")

if __name__ == "__main__":
    export_int8()