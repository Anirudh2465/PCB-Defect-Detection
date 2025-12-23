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
    # Build model from custom yaml (GhostConv version)
    # The 'n' in filename should trigger the 'n' scale from the yaml
    model = YOLO(r"d:\PCB Defect project\yolov12n_ghost.yaml")
    
    # Optional: Load pretrained weights from standard YOLOv12n (transfer learning)
    # This will load matching layers and ignore mismatching ones (Conv -> GhostConv)
    try:
        model.load("yolov12n.pt")
        print("Loaded compatible weights from yolov12n.pt")
    except Exception as e:
        print(f"Could not load weights: {e}")

    # Train
    results = model.train(
        data=r"d:\PCB Defect project\DeepPCB_YOLO\data.yaml", 
        epochs=10, 
        imgsz=640, 
        batch=8, 
        project=r"d:\PCB Defect project\DeepPCB_YOLO\runs",
        name="yolov12n_ghost_pcb",
        exist_ok=True,
        device='cpu' 
    )
    
    # Validate
    metrics = model.val(device='cpu')
    print("mAP50:", metrics.box.map50)

if __name__ == "__main__":
    train()
