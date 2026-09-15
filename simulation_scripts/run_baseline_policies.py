import glob
import importlib.util
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src_scripts.precompute import get_cached_data
from simulation_scripts.run_hilf_naive import calculate_detection_cost_full
import numpy as np


def _load_local_module(module_name, filename):
    module_path = os.path.join(os.path.dirname(__file__), filename)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {module_name} from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate_policy(image_paths, policy_name, beta=0.5):
    n_samples = len(image_paths)
    
    total_cost = 0.0
    offloads = 0
    accepted_cost = 0
    accepted_count = 0
    decision_success = 0
    
    # E2E Accuracy: Success if (Accepted and Y_t == 0) or (Offloaded)
    # Note: Thesis assumes Server (L-ML) is 100% accurate (Oracle)
    e2e_successes = 0

    for t, img_path in enumerate(image_paths):
        cached_data = get_cached_data(img_path)
        # Y_t is the error of the S-ML (1.0 if failure, 0.0 if success)
        Y_t, _, _, _ = calculate_detection_cost_full(cached_data['yolov8n_coco'], cached_data['yolov8x_coco'])
        
        if policy_name == "coin_flip":
            accept_sml = np.random.rand() < 0.5
        elif policy_name == "full_offload":
            accept_sml = False
        elif policy_name == "full_accept":
            accept_sml = True
        elif policy_name == "genie":
            # Theoretical optimum: Accept if correct, Offload if incorrect
            accept_sml = (Y_t == 0)
        else:
            raise ValueError("Unknown policy")

        if accept_sml:
            step_cost = Y_t
            accepted_cost += Y_t
            accepted_count += 1
            if Y_t == 0:
                e2e_successes += 1
        else:
            step_cost = beta
            offloads += 1
            e2e_successes += 1 # Server is Oracle
            
        total_cost += step_cost
        
        # Decision Accuracy: Was the choice optimal?
        # Optimal choice: Accept if Y_t == 0 and Y_t < beta, Offload otherwise.
        # Since Y_t is binary (0 or 1) and beta=0.5:
        # Optimal is: Accept if Y_t=0, Offload if Y_t=1.
        if (Y_t == 1 and not accept_sml) or (Y_t == 0 and accept_sml):
            decision_success += 1

    results = {
        "Policy": policy_name,
        "Decision Accuracy": (decision_success / n_samples) * 100,
        "E2E System Accuracy": (e2e_successes / n_samples) * 100,
        "Local Accuracy": (1.0 - (accepted_cost / accepted_count)) * 100 if accepted_count > 0 else 0.0,
        "Average Cost": total_cost / n_samples,
        "Offload Rate": (offloads / n_samples) * 100
    }
    return results

if __name__ == "__main__":
    limit = 1000 # Using 1000 for faster results, can be 5000 for full set
    image_paths = sorted(glob.glob("datasets/coco_images/val2017/*.jpg"))[:limit]
    
    policies = ["full_accept", "full_offload", "coin_flip", "genie"]
    
    print(f"{'Policy':<15} | {'Decision Acc':<12} | {'E2E Acc':<10} | {'Local Acc':<10} | {'Avg Cost':<8} | {'Offload %':<8}")
    print("-" * 85)
    
    for policy in policies:
        res = evaluate_policy(image_paths, policy)
        print(f"{res['Policy']:<15} | {res['Decision Accuracy']:>11.1f}% | {res['E2E System Accuracy']:>9.1f}% | {res['Local Accuracy']:>9.1f}% | {res['Average Cost']:>8.3f} | {res['Offload Rate']:>8.1f}%")
