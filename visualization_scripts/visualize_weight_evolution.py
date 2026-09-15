
import glob
import os
import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src_scripts.hilf_algo import HIL_F
from simulation_scripts.run_hilf_naive import calculate_detection_cost_full as calculate_detection_cost
from src_scripts.metrics import weakest_link_confidence

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

def visualize():
    # Load history
    print("Loading history...")
    with open('visualization_scripts/weight_evolution_history.pkl', 'rb') as f:
        history = pickle.load(f)

    weight_history = history['weight_history']
    boundary_history = history['boundary_history']
    p_t_history = history['p_t_history']
    y_t_history = history['y_t_history']
    q_t_history = history.get('q_t_history', [])

    fig, ax = plt.subplots(figsize=(12, 7))
    plt.subplots_adjust(bottom=0.25)

    def find_median_p(weights, boundaries):
        """Finds p where the cumulative area is 50% of total area."""
        deltas = np.diff(boundaries)
        areas = np.array(weights) * deltas
        total_area = np.sum(areas)
        if total_area == 0: return 0.5
        target = total_area / 2.0
        
        current_area = 0.0
        for i in range(len(weights)):
            if current_area + areas[i] >= target:
                # Median is in this interval
                needed = target - current_area
                if weights[i] == 0: return boundaries[i]
                return boundaries[i] + (needed / weights[i])
            current_area += areas[i]
        return 1.0

    # Initialize plot elements
    line, = ax.plot([], [], lw=1.5, color='black', alpha=0.4, zorder=4)
    median_line = ax.axvline(x=0, color='black', linestyle='-', lw=2.5, label='Median Decision Boundary', zorder=5)
    current_pt_marker = ax.axvline(x=0, color='lime', linestyle='--', alpha=0.8, label='Sample $p_t$', zorder=6)

    poly_red = None
    poly_blue = None

    status_text = ax.text(0.02, 1.15, '', transform=ax.transAxes, verticalalignment='top', 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), zorder=10)

    def update_plot(t):
        nonlocal poly_red, poly_blue
        t = int(min(t, len(weight_history) - 1))
        weights = weight_history[t]
        boundaries = boundary_history[t]
        
        # 1. Calculate Median
        median_p = find_median_p(weights, boundaries)
        
        # 2. Build step data and split at median for stark contrast
        x_red, y_red = [boundaries[0]], [0]
        x_blue, y_blue = [median_p], [0]
        
        # Final step data for the outline
        x_outline, y_outline = [boundaries[0]], [0]
        
        for i in range(len(weights)):
            b_low = boundaries[i]
            b_high = boundaries[i+1]
            w = weights[i]
            
            # Outline
            x_outline.extend([b_low, b_high])
            y_outline.extend([w, w])
            
            if b_high <= median_p:
                # Entirely in Red region
                x_red.extend([b_low, b_high])
                y_red.extend([w, w])
            elif b_low >= median_p:
                # Entirely in Blue region
                x_blue.extend([b_low, b_high])
                y_blue.extend([w, w])
            else:
                # Interval spans the median - SPLIT IT
                x_red.extend([b_low, median_p])
                y_red.extend([w, w])
                x_blue.extend([median_p, b_high])
                y_blue.extend([w, w])
                
        x_red.append(median_p); y_red.append(0)
        x_blue.append(boundaries[-1]); y_blue.append(0)
        x_outline.append(boundaries[-1]); y_outline.append(0)
        
        # Normalize y for visualization
        max_w = max(weights) if weights else 1
        y_red = [v / max_w for v in y_red]
        y_blue = [v / max_w for v in y_blue]
        y_outline = [v / max_w for v in y_outline]
        
        # 3. Update Fills
        if poly_red is not None and poly_red in ax.collections:
            try:
                poly_red.remove()
            except (ValueError, NotImplementedError):
                pass
        if poly_blue is not None and poly_blue in ax.collections:
            try:
                poly_blue.remove()
            except (ValueError, NotImplementedError):
                pass
        
        poly_red = ax.fill_between(x_red, 0, y_red, color='#d62728', alpha=0.5, label='Probable Offload', zorder=2)
        poly_blue = ax.fill_between(x_blue, 0, y_blue, color='#1f77b4', alpha=0.5, label='Probable Accept', zorder=2)
        
        # Update black outline
        line.set_data(x_outline, y_outline)
        
        # 4. Update Markers
        median_line.set_xdata([median_p, median_p])
        
        if t < len(p_t_history):
            p_val = p_t_history[t]
            y_val = y_t_history[t]
            current_pt_marker.set_xdata([p_val, p_val])
            current_pt_marker.set_visible(True)
            outcome = "Incorrect" if y_val == 1 else "Correct"
            status_text.set_text(f"Sample {t}\n$p_t$: {p_val:.3f}\nMedian Boundary: {median_p:.3f}\nS-ML: {outcome}")
        else:
            current_pt_marker.set_visible(False)
            status_text.set_text(f"End of simulation\nMedian Boundary: {median_p:.3f}")

        ax.set_title(f'HIL-F Weight Distribution Evolution (t={t})')
        fig.canvas.draw_idle()

    ax.set_ylim(0, 1.1)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Confidence ($p_t$)', fontsize=12)
    ax.set_ylabel('Normalized Weight ($w$)', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, which='both', linestyle='--', alpha=0.3)

    # Slider
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    # slider = Slider(ax_slider, 'Sample (t)', 0, len(weight_history) - 1, valinit=0, valfmt='%0.0f')
    slider = Slider(ax_slider, 'Sample (t)', 0, 5000, valinit=0, valfmt='%0.0f')

    def on_slider(val):
        update_plot(int(val))

    slider.on_changed(on_slider)

    update_plot(0)
    print("Interactive visualization with high-contrast Red/Blue regions created.")
    plt.show()

if __name__ == "__main__":
    if not os.path.exists('visualization_scripts/weight_evolution_history.pkl'):
        run_simulation_and_save_history(limit=5000)
    visualize()