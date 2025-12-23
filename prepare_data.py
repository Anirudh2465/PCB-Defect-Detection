import os
import shutil
import glob
from tqdm import tqdm

def convert_bbox(size, box):
    # box: [xmin, ymin, xmax, ymax]
    # return: [x_center, y_center, width, height] normalized
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def process_split(split_name, src_root, dst_root):
    src_img_dir = os.path.join(src_root, split_name, 'images')
    src_lbl_dir = os.path.join(src_root, split_name, 'labels')
    
    dst_img_dir = os.path.join(dst_root, split_name, 'images')
    dst_lbl_dir = os.path.join(dst_root, split_name, 'labels')
    
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)
    
    img_files = glob.glob(os.path.join(src_img_dir, '*.jpg'))
    
    print(f"Processing {split_name} split: {len(img_files)} images found.")
    
    for img_path in tqdm(img_files):
        # Copy image
        file_name = os.path.basename(img_path)
        shutil.copy(img_path, os.path.join(dst_img_dir, file_name))
        
        # Process label
        label_name = file_name.replace('.jpg', '.txt')
        src_label_path = os.path.join(src_lbl_dir, label_name)
        dst_label_path = os.path.join(dst_lbl_dir, label_name)
        
        image_width = 640
        image_height = 640
        
        if os.path.exists(src_label_path):
            with open(src_label_path, 'r') as f_in, open(dst_label_path, 'w') as f_out:
                for line in f_in:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # Original: x_min y_min x_max y_max class_id
                        # Note: Check if coords are absolute or normalized? 
                        # User viewed file 01_PCB__1.txt: "466 441 493 470 3" -> These look like absolute pixels for 640x640
                        
                        xmin = float(parts[0])
                        ymin = float(parts[1])
                        xmax = float(parts[2])
                        ymax = float(parts[3])
                        class_id = int(parts[4])
                        
                        # Fix class_id to be 0-based
                        # Source: 1 to 6. Target: 0 to 5.
                        # However, user's cat output showed class_id 3. 
                        # Let's assume standard mapping: input - 1 for now. 
                        # Wait, let's verify min class id. 
                        # Previous command showed min class 1, max class 6.
                        new_class_id = class_id - 1
                        
                        # Bound check coordinates
                        xmin = max(0, min(xmin, image_width))
                        xmax = max(0, min(xmax, image_width))
                        ymin = max(0, min(ymin, image_height))
                        ymax = max(0, min(ymax, image_height))
                        
                        if xmax <= xmin or ymax <= ymin:
                            continue

                        bb = (xmin, ymin, xmax, ymax)
                        yolo_box = convert_bbox((image_width, image_height), bb)
                        
                        f_out.write(f"{new_class_id} {yolo_box[0]:.6f} {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f}\n")

if __name__ == "__main__":
    src_root = r"d:\PCB Defect project\DeepPCB"
    dst_root = r"d:\PCB Defect project\DeepPCB_YOLO"
    
    # We found subfolders: train, test, valid. 
    # And inside them: images, labels.
    # Actually, previous list_dir showed:
    # DeepPCB/train/images (dir)
    # DeepPCB/train/labels (dir) 
    # So the structure is exactly as expected.
    
    for split in ['train', 'valid', 'test']:  
        if os.path.exists(os.path.join(src_root, split)):
            process_split(split, src_root, dst_root)
        else:
            print(f"Split {split} not found in source.")
            
    print("Data preparation complete.")
