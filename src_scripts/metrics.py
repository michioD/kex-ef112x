import numpy as np


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
        return (1.0 if len(l_boxes) > 0 else 0.0), 0.0, (1.0 if len(l_boxes) > 0 else 0.0), 0.0

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
            continue
            
        # 3. Check if classification matches for the pair
        if s_cls[match_idx] != l_cls[i]:
            misclassified = 1.0
        matched_s_indices.add(match_idx)

    # 4. Check for extra boxes in S-ML
    if len(matched_s_indices) != len(s_xyxy):
        fp = 1.0

    y_t = 1.0 if (fp or fn or misclassified) else 0.0
    return y_t, fp, fn, misclassified


def product_confidence(results):
        """
        Simple product of all object confidences.
        This metric is extremely sensitive to low-confidence detections, making it a strong candidate for high-accuracy targets.
        """
        if not results or len(results[0].boxes) == 0:
            return 0.0
        
        confs = results[0].boxes.conf.cpu().numpy()
        p_t = np.prod(confs)
        return float(np.clip(p_t, 0.0, 1.0))

def weakest_link_confidence(results):
    """
    Image-level confidence based on weakest detected object.
    """
    if not results or len(results[0].boxes) == 0:
        return 0.0

    confs = results[0].boxes.conf.cpu().numpy()

    # Bottleneck principle: weakest object dominates
    p_t = np.min(confs)

    return float(np.clip(p_t, 0.0, 1.0))

# def suppression_metric(results, threshold=0.25):
#     """
#     Measures the ratio of suppressed candidates (< threshold) 
#     to detected objects (>= threshold).
#     """
#     if not results or len(results[0].boxes) == 0:
#         return 0.0
        
#     confs = results[0].boxes.conf.cpu().numpy()
#     num_detected = np.sum(confs >= threshold)
#     num_suppressed = 0
#     for conf in confs:
#         if conf <= threshold and conf > 0.1:
#             num_suppressed += 1
    
#     if num_detected == 0:
#         return float(1.0) # Return raw count if no objects were detected
        
#     return float(num_suppressed / (num_detected+num_suppressed))


def connected_components(nodes, edges):
    unseen = set(nodes)
    adjacency = {node: set() for node in nodes}
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)

    components = []
    while unseen:
        root = unseen.pop()
        stack = [root]
        component = [root]
        while stack:
            node = stack.pop()
            for neighbor in adjacency[node]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
                    component.append(neighbor)
        components.append(component)
    return components

def cluster_raw_detections(results, low_threshold=0.01, iou_threshold=0.5):
    """
    Builds class-wise IoU connected components over low-confidence raw
    detections. Each component is treated as one object-hypothesis cluster.
    """
    if not results or len(results[0].boxes) == 0:
        return []

    boxes_obj = results[0].boxes
    confs = boxes_obj.conf.cpu().numpy()
    keep = confs >= low_threshold
    if not np.any(keep):
        return []

    boxes = boxes_obj.xyxy.cpu().numpy()[keep]
    classes = boxes_obj.cls.cpu().numpy().astype(int)[keep]
    confs = confs[keep]

    clusters = []
    for class_id in np.unique(classes):
        indices = np.where(classes == class_id)[0].tolist()
        edges = []
        for i, idx_a in enumerate(indices):
            for idx_b in indices[i + 1:]:
                if box_iou(boxes[idx_a], boxes[idx_b]) > iou_threshold:
                    edges.append((idx_a, idx_b))

        for component in connected_components(indices, edges):
            component = np.asarray(component, dtype=int)
            cluster_boxes = boxes[component]
            cluster_confs = confs[component]
            top_idx = int(np.argmax(cluster_confs))
            top_box = cluster_boxes[top_idx]
            if len(component) > 1:
                ious_to_top = np.asarray([box_iou(box, top_box) for box in cluster_boxes])
                agreement = float(np.mean(ious_to_top))
                spread = float(np.mean(1.0 - ious_to_top))
            else:
                agreement = 1.0
                spread = 0.0

            clusters.append({
                "size": int(len(component)),
                "max_conf": float(np.max(cluster_confs)),
                "mean_conf": float(np.mean(cluster_confs)),
                "mass": float(np.sum(cluster_confs)),
                "agreement": agreement,
                "spread": spread,
                "class_id": int(class_id),
            })
    return clusters

def calculate_rejected_cluster_mass(results, low_threshold=0.01, final_threshold=0.25, iou_threshold=0.5):
    """
    Sum of max_conf for all clusters that didn't produce a final detection.
    """
    clusters = cluster_raw_detections(
        results,
        low_threshold=low_threshold,
        iou_threshold=iou_threshold,
    )
    if not clusters:
        return 0.0
    
    weak_clusters = [c for c in clusters if c["max_conf"] < final_threshold]
    return float(sum(c["max_conf"] for c in weak_clusters))

def suppression_safety_metric(results, **kwargs):
    """
    Maps unconfirmed signal mass to a [0, 1] safety score.
    p_t = 1 / (1 + alpha * M)
    """
    mass = calculate_rejected_cluster_mass(results, **kwargs)
    return float(1.0 / (1.0 + mass))

