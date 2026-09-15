import torch
import glob
import os
import matplotlib.pyplot as plt
from PIL import Image
from src_scripts.precompute import get_cached_data

def generate_grid(indices, output_path="yolov8_comparison_grid_new.png"):
    image_paths = sorted(glob.glob("datasets/coco_images/val2017/*.jpg"))
    
    rows = len(indices)
    cols = 2 # S-ML vs L-ML
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
    
    for i, idx in enumerate(indices):
        img_path = image_paths[idx]
        cached = get_cached_data(img_path)
        
        s_res = cached['yolov8n_coco'][0]
        l_res = cached['yolov8x_coco'][0]
        
        # Plot using Ultralytics built-in plotter
        # We convert BGR (opencv default) to RGB for matplotlib
        s_plot = s_res.plot()[:, :, ::-1] 
        l_plot = l_res.plot()[:, :, ::-1]
        
        axes[i, 0].imshow(s_plot)
        axes[i, 0].set_title(f"S-ML (YOLOv8n) - Sample {idx}")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(l_plot)
        axes[i, 1].set_title(f"L-ML (YOLOv8x) - Ground Truth")
        axes[i, 1].axis('off')
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved comparison grid to {output_path}")

if __name__ == "__main__":
    # Selected indices that show clear FN improvements
    target_indices = [6, 10, 11]
    generate_grid(target_indices)
