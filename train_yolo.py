from ultralytics import YOLO

def train():
    # Load a model
    # YOLOv11n (nano) is a new smaller model, effectively the successor to v8n
    # If not locally available, it will download.
    model = YOLO("yolo11n.pt") 

    # Train the model
    results = model.train(
        data=r"d:\PCB Defect project\DeepPCB_YOLO\data.yaml", 
        epochs=10, 
        imgsz=640, 
        batch=8, 
        project=r"d:\PCB Defect project\DeepPCB_YOLO\runs",
        name="yolov11_pcb",
        exist_ok=True,
        device='cpu' # Use CPU if no GPU available, or remove to auto-select
    )
    
    # Validate the model
    metrics = model.val(device='cpu')
    print("mAP50:", metrics.box.map50)
    print("mAP50-95:", metrics.box.map)

if __name__ == "__main__":
    train()
