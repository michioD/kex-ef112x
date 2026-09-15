import torch
import glob
import os
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
import numpy as np

def apply_nms_to_results(boxes, scores, classes, conf_threshold=0.25, iou_threshold=0.7):
    # Filter by confidence
    mask = scores >= conf_threshold
    f_boxes = boxes[mask]
    f_scores = scores[mask]
    f_classes = classes[mask]
    
    if len(f_boxes) == 0:
        return np.array([]), np.array([]), np.array([])
        
    # Apply NMS
    keep = torchvision.ops.batched_nms(
        torch.from_numpy(f_boxes), 
        torch.from_numpy(f_scores), 
        torch.from_numpy(f_classes), 
        iou_threshold
    )
    
    # Convert to numpy and handle single-item case explicitly
    idx = keep.cpu().numpy()
    return f_boxes[idx], f_scores[idx], f_classes[idx]

def visualize_index(img_path, output_path="raw_vs_nms_vis.png"):
    image_paths = sorted(glob.glob("datasets/coco_images/val2017/*.jpg"))
    # img_path = image_paths[idx]
    
    print(f"Visualizing index {img_path}")
    
    # Load raw data
    base_name = os.path.basename(img_path)
    cache_file = os.path.join("cache_yolo_results_raw", f"{base_name}.pt")
    cached_raw = torch.load(cache_file, map_location="cpu", weights_only=False)
    sml_raw = cached_raw['yolov8n_coco_raw'][0]
    
    raw_boxes = sml_raw.boxes.xyxy.cpu().numpy()
    raw_confs = sml_raw.boxes.conf.cpu().numpy()
    raw_cls = sml_raw.boxes.cls.cpu().numpy()
    
    # Get Post-NMS boxes (Purple)
    nms_boxes, nms_confs, _ = apply_nms_to_results(raw_boxes, raw_confs, raw_cls)
    
    # Get All Raw boxes above 0.01 (Yellow)
    raw_mask = raw_confs > 0.01
    vis_raw_boxes = raw_boxes[raw_mask]
    vis_raw_confs = raw_confs[raw_mask]

    img = Image.open(img_path)
    plt.figure(figsize=(15, 15))
    plt.imshow(img)
    ax = plt.gca()
    
    # 1. Draw ALL RAW hypotheses in Yellow
    for box, conf in zip(vis_raw_boxes, vis_raw_confs):
        alpha = max(0.05, min(conf * 1.5, 0.4))
        rect = plt.Rectangle(
            (box[0], box[1]), 
            box[2]-box[0], 
            box[3]-box[1], 
            fill=False, 
            edgecolor='yellow', 
            linewidth=0.5, 
            alpha=alpha
        )
        ax.add_patch(rect)
        
    # 2. Draw FINAL NMS Detections in Purple
    for box, conf in zip(nms_boxes, nms_confs):
        rect = plt.Rectangle(
            (box[0], box[1]), 
            box[2]-box[0], 
            box[3]-box[1], 
            fill=False, 
            edgecolor='purple', 
            linewidth=3.0, 
            alpha=0.9
        )
        ax.add_patch(rect)
        plt.text(box[0], box[1]-5, f"{conf:.2f}", color='purple', weight='bold', fontsize=12, backgroundcolor='white')

    plt.title(f"SML Comparison: Raw Hypotheses (Yellow) vs. Final Detections (Purple)\nImage: {os.path.basename(img_path)}")
    plt.axis('off')
    
    # Add legend manually
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='yellow', lw=2, label='Raw Hypotheses (Suppressed/Noise)'),
        Line2D([0], [0], color='purple', lw=4, label='Final Detections (Passed NMS)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)

    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, default="datasets/coco_images/val2017/000000012120.jpg")
    parser.add_argument("--output", type=str, default="raw_vs_nms_vis.png")
    args = parser.parse_args()
    visualize_index(args.img_path, args.output)
