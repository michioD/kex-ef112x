import torch
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from src_scripts.precompute import get_cached_data

def visualize_standard_detections(limit=10, output_dir="figures/detections"):
    image_paths = sorted(glob.glob("datasets/coco_images/val2017/*.jpg"))[:limit]
    
    print(f"Visualizing standard detections for first {limit} images...")
    
    for i, img_path in enumerate(image_paths):
        base_name = os.path.basename(img_path)
        output_path = os.path.join(output_dir, f"{os.path.splitext(base_name)[0]}_det.png")
        
        # Load standard cached data (post-NMS)
        cached_data = get_cached_data(img_path)
        sml = cached_data['yolov8n_coco'][0]
        
        boxes = sml.boxes.xyxy.cpu().numpy()
        confs = sml.boxes.conf.cpu().numpy()
        classes = sml.boxes.cls.cpu().numpy()
        
        img = Image.open(img_path)
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(img)
        
        for box, conf, cls in zip(boxes, confs, classes):
            x1, y1, x2, y2 = map(float, box)
            w, h = x2 - x1, y2 - y1
            rect = patches.Rectangle(
                (x1, y1), w, h, 
                fill=False, edgecolor='cyan', linewidth=2.0
            )
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, f"{int(cls)}:{conf:.2f}", color='white', weight='bold', 
                    fontsize=10, bbox=dict(facecolor='cyan', alpha=0.7, pad=1))

        ax.set_title(f"YOLOv8n Detections (Standard Post-NMS)\n{base_name}")
        ax.axis('off')
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {output_path}")

if __name__ == "__main__":
    visualize_standard_detections()
