import sys
import os

repo_path = r"d:\PCB Defect project\yolov12_repo"
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

try:
    from ultralytics import YOLO
except ImportError:
    print("Error importing Ultralytics")
    sys.exit(1)

def train():
    model = YOLO(r"d:\PCB Defect project\yolov12n_ghost_extreme.yaml")
    
    # Try loading weights, ignore mismatch
    try:
        model.load("yolov12n.pt")
        print("Loaded compatible weights from yolov12n.pt")
    except Exception as e:
        print(f"Could not load weights: {e}")

    results = model.train(
        data=r"d:\PCB Defect project\DeepPCB_YOLO\data.yaml", 
        epochs=10, 
        imgsz=640, 
        batch=8, 
        project=r"d:\PCB Defect project\DeepPCB_YOLO\runs",
        name="yolov12n_ghost_extreme_pcb",
        exist_ok=True,
        device='cpu' 
    )
    
    metrics = model.val(device='cpu')
    print("mAP50:", metrics.box.map50)

if __name__ == "__main__":
    train()
