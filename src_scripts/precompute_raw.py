import os
import glob
import torch
from ultralytics import YOLO
from tqdm import tqdm

# ============================================================
# Configuration
# ============================================================
CACHE_DIR = "cache_yolo_results_raw"
IMAGE_DIR = "datasets/coco_images/val2017/*.jpg"


def resolve_device():
    """Choose the best available inference backend without assuming Apple Silicon."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_precompute():
    """
    Generates and saves raw YOLO predictions for all images.
    Internal NMS is effectively disabled (iou=1.0) and the confidence 
    threshold is lowered (conf=0.001) to cache all possible bounding boxes.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    image_paths = sorted(glob.glob(IMAGE_DIR))
    
    # Initialize models only when precomputing is necessary
    device = resolve_device()
    print(f"Using {device} for inference")

    s_ml = YOLO('yolov8n.pt')
    l_ml = YOLO('yolov8x.pt')
    s_ml.to(device)
    l_ml.to(device)

    for img_path in tqdm(image_paths, desc="Precomputing raw YOLO predictions"):
        base_name = os.path.basename(img_path)
        cache_file = os.path.join(CACHE_DIR, f"{base_name}.pt")
        
        # Skip if this specific image has already been processed
        if os.path.exists(cache_file):
            continue
            
        # Bypass NMS and default confidence thresholds to cache raw predictions
        sml_results = s_ml.predict(
            img_path, 
            conf=0.01, 
            iou=1.0, 
            max_det=10000, 
            verbose=False
        )
        lml_results = l_ml.predict(
            img_path, 
            conf=0.01, 
            iou=1.0, 
            max_det=10000, 
            verbose=False
        )
        
        # Save the results list directly to disk
        torch.save({
            'yolov8n_coco_raw': sml_results,
            'yolov8x_coco_raw': lml_results
        }, cache_file)

def check_cache():
    """
    Verifies the cache directory exists and contains the correct number of files.
    """
    image_paths = sorted(glob.glob(IMAGE_DIR))
    
    if not os.path.exists(CACHE_DIR) or len(os.listdir(CACHE_DIR)) < len(image_paths):
        print("Cache incomplete or missing. Initializing precomputation...")
        run_precompute()
    else:
        print("Cache fully populated.")
    
def get_cached_data_raw(img_path):
    """
    Retrieves the cached YOLO results for a specific image.
    """
    base_name = os.path.basename(img_path)
    cache_file = os.path.join(CACHE_DIR, f"{base_name}.pt")
    
    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"Cache file not found for {img_path}. Please run precompute first.")
        
    return torch.load(cache_file, map_location="cpu", weights_only=False)

if __name__ == "__main__":
    # Run the cache check directly if this script is executed
    check_cache()
