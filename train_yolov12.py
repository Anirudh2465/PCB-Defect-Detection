import sys
import os

# Insert the cloned repo path at the beginning of sys.path to use its ultralytics version
repo_path = r"d:\PCB Defect project\yolov12_repo"
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

try:
    from ultralytics import YOLO
    print(f"Loaded Ultralytics from: {os.path.dirname(os.path.dirname(YOLO.__module__))}")
except ImportError as e:
    print(f"Error importing Ultralytics: {e}")
    sys.exit(1)

def train():
    # Load YOLOv12 model
    # We will try 'yolov12n.pt'. If it doesn't download automatically, 
    # we might need to specify the yaml config to build from scratch.
    # Usually 'yolov12n.yaml' works if weights are missing, but let's try .pt first.
    try:
        model = YOLO("yolov12n.pt") 
    except Exception:
        print("Could not load yolov12n.pt, trying yolov12n.yaml (scratch build)")
        # The yaml file is in ultralytics/cfg/models/v12/yolov12.yaml
        # But we need to define the size variant. Typically handled by kwargs or separate yamls.
        # The repo only had yolov12.yaml. Let's inspect passing scale='n' if needed 
        # or just "yolov12.yaml" and hope for the best default.
        model = YOLO(r"d:\PCB Defect project\yolov12_repo\ultralytics\cfg\models\v12\yolov12.yaml")

    # Train the model
    # Using CPU because of GPU compatibility issues (RTX 5070)
    results = model.train(
        data=r"d:\PCB Defect project\DeepPCB_YOLO\data.yaml", 
        epochs=10, 
        imgsz=640, 
        batch=8, 
        project=r"d:\PCB Defect project\DeepPCB_YOLO\runs",
        name="yolov12_pcb",
        exist_ok=True,
        device='cpu' 
    )
    
    # Validate
    metrics = model.val(device='cpu')
    print("mAP50:", metrics.box.map50)
    print("mAP50-95:", metrics.box.map)

if __name__ == "__main__":
    train()
