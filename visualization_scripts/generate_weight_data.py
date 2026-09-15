import os
import glob
import torch
import numpy as np
import pickle
from src_scripts.hilf_algo import HIL_F
from src_scripts.metrics import *
from simulation_scripts.run_hilf_naive import calculate_detection_cost_full as calculate_detection_cost

CACHE_DIR = "cache_yolo_results"
IMAGE_DIR = "datasets/coco_images/val2017/*.jpg"

confidence_metric = weakest_link_confidence

def run_simulation_and_save_history(limit=5000):
    image_paths = sorted(glob.glob(IMAGE_DIR))[:limit]
    n_samples = len(image_paths)
    hil_f = HIL_F(n_samples=n_samples)
    
    print(f"Running simulation on {n_samples} samples...")
    for t, img_path in enumerate(image_paths):
        base_name = os.path.basename(img_path)
        cache_file = os.path.join(CACHE_DIR, f"{base_name}.pt")
        
        if not os.path.exists(cache_file):
            continue
            
        cached = torch.load(cache_file, weights_only=False)
        sml = cached['yolov8n_coco']
        lml = cached['yolov8x_coco']
        
        p_t = confidence_metric(sml)
        y_t = calculate_detection_cost(sml, lml)[0]
        
        hil_f.get_decision(p_t) # Just to advance state if needed
        hil_f.update(p_t, y_t)
        
        if (t + 1) % 500 == 0:
            print(f"Processed {t+1}/{n_samples}...")
            
    history_data = {
        'weight_history': hil_f.weight_history,
        'boundary_history': hil_f.boundary_history,
        'p_t_history': [h[0] for h in hil_f.history],
        'y_t_history': [h[1] for h in hil_f.history]
    }
    
    with open('visualization_scripts/weight_evolution_history.pkl', 'wb') as f:
        pickle.dump(history_data, f)
    print("History saved to weight_evolution_history.pkl")

if __name__ == "__main__":
    run_simulation_and_save_history(limit=5000)
