
import os
import glob
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src_scripts.metrics import weakest_link_confidence, calculate_detection_cost_full

CACHE_DIR = "cache_yolo_results"

def gather_data(limit=1000):
    cache_files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.pt")))[:limit]
    data = []
    
    print(f"Gathering data for {len(cache_files)} samples...")
    
    for i, cache_file in enumerate(cache_files):
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(cache_files)}...")
            
        cached = torch.load(cache_file, weights_only=False)
        sml = cached['yolov8n_coco']
        lml = cached['yolov8x_coco']
        
        pt = weakest_link_confidence(sml)
        cost = calculate_detection_cost_full(sml, lml)[0]
        is_correct = (cost == 0.0)
        
        data.append({
            'pt': pt,
            'is_correct': is_correct
        })
        
    return pd.DataFrame(data)

def plot_weakest_link_distribution(output_path="figures/weakest_link_dist.png"):
    df = gather_data(limit=5000)
    sns.set_theme(style="white")
    
    plt.figure(figsize=(10, 6))
    
    # Bins for p_t
    bins = np.linspace(0, 1, 21) # 0.05 width bins
    
    correct = df[df['is_correct'] == True]['pt']
    incorrect = df[df['is_correct'] == False]['pt']
    
    # We want them stacked to see the total and the composition.
    plt.hist([correct, incorrect], bins=bins, stacked=True, 
             label=['Correct Inferences', 'Incorrect Inferences'],
             color=['#2ca02c', '#d62728'], alpha=0.7, rwidth=0.85, edgecolor='black')
    
    plt.xlabel('Confidence Score ($p_t$)', fontsize=14)
    plt.ylabel('Number of Images', fontsize=14)
    plt.title('Distribution of Correct vs. Incorrect Inferences (Weakest Link)', fontsize=16)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    plot_weakest_link_distribution(output_path="figures/weakest_link_dist.png")
