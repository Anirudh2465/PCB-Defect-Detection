import os
import glob
import random
import cv2
import numpy as np
from ultralytics import YOLO

# Class names mapping
CLASS_NAMES = ['open', 'short', 'mousebite', 'spur', 'copper', 'pin-hole']

def get_box_coords(box, img_w, img_h):
    # YOLO format: class x_c y_c w h (normalized)
    # Return: xcycwh in pixels
    # But for drawing we often want xyxy (top-left, bottom-right)
    # Let's return standard xyxy for cv2.rectangle
    
    xc, yc, w, h = box[1], box[2], box[3], box[4]
    
    x1 = int((xc - w/2) * img_w)
    y1 = int((yc - h/2) * img_h)
    x2 = int((xc + w/2) * img_w)
    y2 = int((yc + h/2) * img_h)
    
    return x1, y1, x2, y2

def run_inference():
    weights_path = r"d:\PCB Defect project\DeepPCB_YOLO\runs\yolov11_pcb\weights\best.pt"
    if not os.path.exists(weights_path):
        weights_path = r"d:\PCB Defect project\DeepPCB_YOLO\runs\yolov11_pcb\weights\last.pt"
    
    print(f"Loading model from {weights_path}...")
    model = YOLO(weights_path)

    test_img_dir = r"d:\PCB Defect project\DeepPCB_YOLO\test\images"
    test_lbl_dir = r"d:\PCB Defect project\DeepPCB_YOLO\test\labels"
    output_dir = r"d:\PCB Defect project\DeepPCB_YOLO\inference_results"
    os.makedirs(output_dir, exist_ok=True)

    all_images = glob.glob(os.path.join(test_img_dir, "*.jpg"))
    if not all_images:
        print("No test images found.")
        return

    # Process 10 random images
    sample_images = random.sample(all_images, min(len(all_images), 10))

    print(f"Running inference on {len(sample_images)} images...")
    
    for img_path in sample_images:
        # Load image
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        
        # 1. DRAW GROUND TRUTH (Green)
        label_name = os.path.basename(img_path).replace('.jpg', '.txt')
        label_path = os.path.join(test_lbl_dir, label_name)
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = list(map(float, line.strip().split()))
                    cls_id = int(parts[0])
                    # parts: [class, xc, yc, w, h]
                    x1, y1, x2, y2 = get_box_coords(parts, w, h)
                    
                    # Draw GT box - Green
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label_text = f"GT: {CLASS_NAMES[cls_id]}"
                    cv2.putText(img, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 2. DRAW PREDICTIONS (Red)
        results = model(img_path, device='cpu')
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # box.xyxy provides pixel coordinates [x1, y1, x2, y2]
                coords = box.xyxy[0].cpu().numpy() # [x1, y1, x2, y2]
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                x1, y1, x2, y2 = map(int, coords)
                
                # Draw Pred box - Red
                # Offset slightly to not completely overlap if perfect match
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label_text = f"Pred: {CLASS_NAMES[cls_id]} {conf:.2f}"
                # Put text slightly below or inside
                cv2.putText(img, label_text, (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Save result
        base_name = os.path.basename(img_path)
        save_path = os.path.join(output_dir, "v11_" + base_name)
        cv2.imwrite(save_path, img)
        print(f"Saved comparison: {save_path}")

    print("Inference comparison complete.")

if __name__ == "__main__":
    run_inference()
