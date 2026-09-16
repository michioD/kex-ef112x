import csv
import os
import sys
import numpy as np
import glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src_scripts.metrics import *
from src_scripts.precompute import get_cached_data
from src_scripts.precompute_raw import get_cached_data_raw
from src_scripts.hilf_algo import HIL_F

CACHE_DIR = "cache_yolo_results"


def box_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0



def calculate_detection_cost_full(sml_results, lml_results, iou_threshold=0.45):
    """
    Step-by-step Strict Cost Function:
    1. Cardinality check (N_s == N_l)
    2. Spatial matching for every L-ML box to a unique S-ML box
    3. Classification check for every matched pair
    4. Verify no extra/unmatched boxes in S-ML
    """
    s_boxes = sml_results[0].boxes
    l_boxes = lml_results[0].boxes
    Y_t = 0.0
    fp = 0.0
    fn = 0.0
    misclassified = 0.0
    
    # 1. Cardinality Check
    if len(s_boxes) != len(l_boxes):
        if len(s_boxes) > len(l_boxes):
            fp = 1.0
        else:            
            fn = 1.0
    
    # Base case: both empty is a success
    if len(s_boxes) == 0:
        return 0.0, 0.0, 0.0, 0.0

    s_xyxy = s_boxes.xyxy.cpu().numpy()
    s_cls = s_boxes.cls.cpu().numpy()
    l_xyxy = l_boxes.xyxy.cpu().numpy()
    l_cls = l_boxes.cls.cpu().numpy()

    matched_s_indices = set()

    # 2. Find a matching for each box in L-ML
    for i in range(len(l_xyxy)):
        best_iou = -1.0
        match_idx = -1
        for j in range(len(s_xyxy)):
            if j in matched_s_indices:
                continue
            iou = box_iou(l_xyxy[i], s_xyxy[j])
            if iou > best_iou:
                best_iou = iou
                match_idx = j

        # Check if spatial match exists
        if match_idx == -1 or best_iou < iou_threshold:
            fn = 1.0
            
        # 3. Check if classification matches for the pair
        if s_cls[match_idx] != l_cls[i]:
            misclassified = 1.0
        matched_s_indices.add(match_idx)

    # 4. Check for extra boxes in S-ML
    if len(matched_s_indices) != len(s_xyxy):
        fp = 1.0

    Y_t = fp or fn or misclassified
    return Y_t, fp, fn, misclassified



confidence_metric = weakest_link_confidence

def run_hierarchical_inference_simulation(
    image_paths,
    output_csv="results/hilf_results_dual_gate_same_yt.csv"):
    n_samples = len(image_paths)
    beta = 0.5
    hil_f_weakest = HIL_F(n_samples=n_samples, beta=beta)
    hil_f_ssm = HIL_F(n_samples=n_samples, beta=beta)
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Index",
            "p_t",
            "q_t",
            "cluster_safety_t",
            "q_cluster_t",
            "Action",
            "Y_t",
            "Success",
            "FP",
            "FN",
            "Misclassified",
            "SSM decision",
            "Weakest Link decision"
        ])


    total_cost = 0.0
    offloads = 0
    accepted_cost = 0
    accepted_count = 0
    correct_identifications = 0
    decision_success = 0
    correct_offload = 0
    incorrect_offload = 0
    correct_accept = 0
    incorrect_accept = 0
    weakest_only_accepts = 0
    ssm_only_accepts = 0
    both_gate_rejects = 0

    for t, img_path in enumerate(image_paths):
        cached_data = get_cached_data(img_path)
        cached_data_raw = get_cached_data_raw(img_path)
        Y_t, fp, fn, misclassified = calculate_detection_cost_full(cached_data['yolov8n_coco'], cached_data['yolov8x_coco'])
        p_t = confidence_metric(cached_data['yolov8n_coco'])
        s_t = suppression_safety_metric(cached_data_raw['yolov8n_coco_raw'])

        
        accept_sml_classification, q_t = hil_f_weakest.get_decision(p_t)
        accept_sml_cluster, q_t_cluster = hil_f_ssm.get_decision(s_t)

        accept_sml = accept_sml_classification and accept_sml_cluster
        weakest_only_accepts += int(accept_sml_classification and not accept_sml_cluster)
        ssm_only_accepts += int(accept_sml_cluster and not accept_sml_classification)
        both_gate_rejects += int(not accept_sml_classification and not accept_sml_cluster)

        if accept_sml:
            step_cost = Y_t
            action = "ACCEPTED"
            accepted_cost += Y_t
            accepted_count += 1
            if Y_t == 0:
                correct_identifications += 1
        elif not accept_sml:
            step_cost = beta
            offloads += 1
            action = "OFFLOADED"
            correct_identifications += 1 # Assumed correct via Oracle L-ML

        total_cost += step_cost

        # update with same Y_t for both gates 
        hil_f_weakest.update(p_t, Y_t)
        hil_f_ssm.update(s_t, Y_t)
        

        decision_success += 1 if (Y_t > 0 and not accept_sml) or (Y_t == 0 and accept_sml) else 0

        correct_accept += 1 if accept_sml and Y_t == 0 else 0
        incorrect_accept += 1 if accept_sml and Y_t > 0 else 0
        correct_offload += 1 if not accept_sml and Y_t > 0 else 0
        incorrect_offload += 1 if not accept_sml and Y_t == 0 else 0
        success = "Yes" if Y_t == 0 else "No"
        
        if (t + 1) % 100 == 0 or t == 0:
            print(
                f"Sample {t+1:04d} | p_t: {p_t:.3f} | q_t: {q_t:.3f} | "
                f"cluster_safety: {s_t:.3f} | q_cluster: {q_t_cluster:.3f} | "
                f"Action: {action:<9} | Cost: {Y_t} | Success: {success}"
            )
            
        with open(output_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([t, p_t, q_t, s_t, q_t_cluster, action, Y_t, success, fp, fn, misclassified])


    print("\n--- Simulation Complete ---")
    print(f"Overall System Accuracy: {((correct_accept + correct_offload + incorrect_offload)/n_samples)*100:.1f}%")
    if accepted_count > 0:
        avg_acc = 1.0 - (accepted_cost / accepted_count)
        print(f"Accepted Sample Accuracy (Local): {avg_acc*100:.1f}%")
    print(f"Offloading Rate: {(offloads/n_samples)*100:.1f}%")
    
    if n_samples > 0:
        decision_accuracy = (decision_success / n_samples) * 100
        print(f"Decision Accuracy (Optimal Choice): {decision_accuracy:.1f}%")
    
    # average cost per sample
    avg_cost = total_cost / n_samples if n_samples > 0 else 0.
    print(f"Average Cost per Sample: {avg_cost:.3f}")

    # number of images correctly offloaded
    print(f"Correct Offloads: {correct_offload}")
    # number of images incorrectly offloaded
    print(f"Incorrect Offloads: {incorrect_offload}")
    # number of images correctly accepted
    print(f"Correct Accepts: {correct_accept}")
    # number of images incorrectly accepted
    print(f"Incorrect Accepts: {incorrect_accept}")
    print(f"Base line sml accuracy: {(correct_accept + incorrect_offload) / n_samples * 100:.1f}%")
    print(f"Weakest accepted, ssm rejected: {weakest_only_accepts}")
    print(f"SSM accepted, weakest rejected: {ssm_only_accepts}")
    print(f"Both gates rejected: {both_gate_rejects}")


if __name__ == "__main__":
    image_paths = sorted(glob.glob("datasets/coco_images/val2017/*.jpg"))[0:5000]
    run_hierarchical_inference_simulation(image_paths)
