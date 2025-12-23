import sys
import os
import glob
import random
import cv2
import numpy as np

# Insert the cloned repo path at the beginning of sys.path
repo_path = r"d:\PCB Defect project\yolov12_repo"
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

from ultralytics import YOLO

# Class names mapping
CLASS_NAMES = ['open', 'short', 'mousebite', 'spur', 'copper', 'pin-hole']

def get_box_coords(box, img_w, img_h):
    xc, yc, w, h = box[1], box[2], box[3], box[4]
    x1 = int((xc - w/2) * img_w)
    y1 = int((yc - h/2) * img_h)
    x2 = int((xc + w/2) * img_w)
    y2 = int((yc + h/2) * img_h)
    return x1, y1, x2, y2

def run_inference():
    weights_path = r"d:\PCB Defect project\DeepPCB_YOLO\runs\yolov12_pcb\weights\best.pt"
    if not os.path.exists(weights_path):
        weights_path = r"d:\PCB Defect project\DeepPCB_YOLO\runs\yolov12_pcb\weights\last.pt"
    
    if not os.path.exists(weights_path):
        print("Weights not found. Wait for training to complete.")
        return

    print(f"Loading YOLOv12 model from {weights_path}...")
    model = YOLO(weights_path)

    test_img_dir = r"d:\PCB Defect project\DeepPCB_YOLO\test\images"
    test_lbl_dir = r"d:\PCB Defect project\DeepPCB_YOLO\test\labels"
    output_dir = r"d:\PCB Defect project\DeepPCB_YOLO\inference_results_v12"
    os.makedirs(output_dir, exist_ok=True)

    all_images = glob.glob(os.path.join(test_img_dir, "*.jpg"))
    if not all_images:
        print("No test images found.")
        return

    sample_images = random.sample(all_images, min(len(all_images), 10))
    print(f"Running inference on {len(sample_images)} images...")
    
    for img_path in sample_images:
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        
        # 1. GT (Green)
        label_name = os.path.basename(img_path).replace('.jpg', '.txt')
        label_path = os.path.join(test_lbl_dir, label_name)
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = list(map(float, line.strip().split()))
                    cls_id = int(parts[0])
                    x1, y1, x2, y2 = get_box_coords(parts, w, h)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f"GT: {CLASS_NAMES[cls_id]}", (x1, y1 - 5), 0, 0.5, (0, 255, 0), 1)
        
        # 2. Pred (Red)
        results = model(img_path, device='cpu')
        for result in results:
            for box in result.boxes:
                coords = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, coords)
                
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, f"Pred: {CLASS_NAMES[cls_id]} {conf:.2f}", (x1, y2 + 15), 0, 0.5, (0, 0, 255), 1)

        base_name = os.path.basename(img_path)
        save_path = os.path.join(output_dir, "v12_" + base_name)
        cv2.imwrite(save_path, img)
        print(f"Saved: {save_path}")

if __name__ == "__main__":
    run_inference()
