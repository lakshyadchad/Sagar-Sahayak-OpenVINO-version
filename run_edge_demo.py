import openvino as ov
import cv2
import numpy as np
import time
from ultralytics import YOLO  # type: ignore

# --- CONFIGURATION (Use r"" for safe Windows paths) ---
ENHANCER_MODEL = r""  #xml
DETECTOR_MODEL = r""  #xml
VIDEO_SOURCE = r""

# --- HELPER: CONTRAST BOOSTER (CLAHE) ---
def contrast_booster(image_rgb):
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return final

def run_demo():
    print("--- Starting OpenVINO Edge Demo ---")

    # 1. CHECK IF FILES EXIST (Prevents instant crash)
    import os
    if not os.path.exists(VIDEO_SOURCE):
        print(f"\nERROR: Video file not found at:\n{VIDEO_SOURCE}\n")
        input("Press Enter to exit...")
        return

    # 2. INITIALIZE OPENVINO
    core = ov.Core()

    print("Loading Enhancer...")
    model_enhancer = core.read_model(model=ENHANCER_MODEL)
    compiled_enhancer = core.compile_model(model=model_enhancer, device_name="CPU")
    infer_request = compiled_enhancer.create_infer_request()
    input_layer = compiled_enhancer.input(0)
    output_layer = compiled_enhancer.output(0)

    print("Loading Detector...")
    detector = YOLO(DETECTOR_MODEL, task='detect')

    # 3. START VIDEO (Standard Capture - No Threading)
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    print("System Running. Press 'q' to exit.")
    
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        
        # Loop video if it ends (Optional - keeps window open)
        if not ret: 
            print("Video ended. Restarting...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # --- STEP A: PRE-PROCESSING ---
        # 1. Resize to 256x256
        img_small = cv2.resize(frame, (256, 256))
        
        # 2. Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
        
        # 3. Normalize (MEAN=0.5, STD=0.5) 
        img_float = img_rgb.astype(np.float32) / 255.0
        img_normalized = (img_float - 0.5) / 0.5
        
        # 4. Prepare Tensor (CHW)
        input_tensor = img_normalized.transpose(2, 0, 1)
        input_tensor = np.expand_dims(input_tensor, 0)

        # --- STEP B: INFERENCE (ENHANCER) ---
        res = infer_request.infer({input_layer: input_tensor})
        result_tensor = res[output_layer]

        # --- STEP C: POST-PROCESSING ---
        # 1. Transpose back to HWC
        clean_out = result_tensor[0].transpose(1, 2, 0)
        
        # 2. Denormalize: (x + 1) / 2
        clean_out = (clean_out + 1.0) / 2.0
        
        # 3. Clip and Convert
        clean_out = np.clip(clean_out, 0.0, 1.0)
        clean_uint8 = (clean_out * 255).astype(np.uint8)

        # 4. Apply CLAHE
        final_enhanced_rgb = contrast_booster(clean_uint8)

        # --- STEP D: UPSCALE FOR YOLO ---
        # Resize 256x256 -> 1024x1024 (Bicubic)
        yolo_input_rgb = cv2.resize(final_enhanced_rgb, (1024, 1024), interpolation=cv2.INTER_CUBIC)
        yolo_input_bgr = cv2.cvtColor(yolo_input_rgb, cv2.COLOR_RGB2BGR)

        # --- STEP E: DETECTION ---
        results = detector(yolo_input_bgr, verbose=False, conf=0.25)
        annotated_frame = results[0].plot()

        # --- STEP F: VISUALIZATION ---
        # Resize for display (e.g., 640x480)
        display_w, display_h = 640, 480
        
        view_raw = cv2.resize(frame, (display_w, display_h))
        view_enhanced = cv2.resize(annotated_frame, (display_w, display_h))

        # Stitch Side-by-Side
        final_display = cv2.hconcat([view_raw, view_enhanced])

        # FPS Calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time

        # Labels
        cv2.putText(final_display, "RAW INPUT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(final_display, "ENHANCED + AI", (display_w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(final_display, f"FPS: {fps:.1f}", (10, final_display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Intel OpenVINO Edge Demo", final_display)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_demo()