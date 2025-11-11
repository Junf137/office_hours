"""
Utility functions for evaluation and metric calculations in spatial association experiments.
"""
import os
import json
import difflib
from typing import List, Dict


def load_gt_for_episode(i: int, ground_truth_dir: str, split: bool = False) -> Dict:
    """
    Load ground truth data for a specific episode.
    
    Args:
        i: Episode index
        ground_truth_dir: Path to directory containing ground truth files
        split: Whether to use split videos and ground truth
        
    Returns:
        Dictionary containing ground truth data
    """
    if split:
        path = os.path.join(ground_truth_dir, f"episode_0_gt_part_{i}.json")
    else:
        path = os.path.join(ground_truth_dir, f"episode_{i}_gt.json")
    if not os.path.exists(path):
        return {}
    return json.load(open(path, "r"))


def mape_count(gt_count: int, pred_count: int) -> float:
    """
    Mean Absolute Percentage Error for single scalar counts.
    
    Args:
        gt_count: Ground truth count
        pred_count: Predicted count
        
    Returns:
        MAPE value as percentage
    """
    # handle zero gt case
    if gt_count == 0:
        return 0.0 if pred_count == 0 else 100.0
    # compute |(pred-gt)/gt| * 100
    return abs(pred_count - gt_count) / gt_count * 100.0


def _name_sim(a: str, b: str) -> float:
    """
    Calculate similarity between two name strings using SequenceMatcher.
    
    Args:
        a: First name string
        b: Second name string
        
    Returns:
        Similarity ratio between 0 and 1
    """
    if a is None or b is None:
        return 0.0
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def build_internal_id_mapping(
    parsed_response: Dict,
    ground_truth: Dict,
    name_threshold: float = 0.85
) -> Dict[str, Dict]:
    """
    Build mapping from internal_id to ground truth cubicle.
    
    Strategy:
    1. Try exact ID match (if readable)
    2. Try name fuzzy match (if readable)
    3. If both unreadable/noID, mark as "unmatched" but preserve in mapping
    
    Args:
        parsed_response: Model output with internal_ids
        ground_truth: Ground truth data
        name_threshold: Similarity threshold for name matching
        
    Returns:
        Dict mapping internal_id -> {
            "gt_cubicle": matched GT cubicle or None,
            "match_type": "id" | "name" | "unmatched",
            "pred_cubicle": the predicted cubicle data
        }
    """
    mapping = {}
    gt_cubicles = ground_truth.get("cubicles", [])
    gt_used = [False] * len(gt_cubicles)  # Track which GT cubicles have been matched
    
    pred_cubicles = parsed_response.get("cubicles", [])
    
    ## Try ID matches first (highest confidence)
    for pred_cubicle in pred_cubicles:
        internal_id = pred_cubicle["internal_id"]
        pred_id = pred_cubicle["id"]
        pred_name = pred_cubicle["name"]
        
        matched_gt = None
        match_type = "unmatched"
        
        # Try exact ID match (if ID is readable and not noID)
        if pred_id not in ["Unreadable", "noID", ""]:
            for idx, gt_cubicle in enumerate(gt_cubicles):
                if not gt_used[idx] and gt_cubicle["id"] == pred_id:
                    matched_gt = gt_cubicle
                    match_type = "id"
                    gt_used[idx] = True
                    break
        
        mapping[internal_id] = {
            "gt_cubicle": matched_gt,
            "match_type": match_type,
            "pred_cubicle": pred_cubicle,
        }

    ## Try name matches for remaining unmatched predictions
    for internal_id, match_info in mapping.items():
        if match_info["match_type"] != "unmatched":
            continue
        
        pred_cubicle = match_info["pred_cubicle"]
        pred_name = pred_cubicle["name"]
        
        # Try name matching (if name is readable)
        if pred_name not in ["Unknown", "Unreadable", ""]:
            best_match_idx = None
            best_similarity = 0.0
            
            for idx, gt_cubicle in enumerate(gt_cubicles):
                if gt_used[idx]:
                    continue
                
                gt_name = gt_cubicle["name"]
                similarity = _name_sim(pred_name, gt_name)
                
                if similarity >= name_threshold and similarity > best_similarity:
                    best_match_idx = idx
                    best_similarity = similarity
            
            if best_match_idx is not None:
                match_info["gt_cubicle"] = gt_cubicles[best_match_idx]
                match_info["match_type"] = "name"
                gt_used[best_match_idx] = True
    
    return mapping


def compute_neighbor_metrics(
    parsed_response: Dict,
    ground_truth: Dict,
    name_threshold: float = 0.85
) -> Dict:
    """
    Compute neighbor relationship metrics.
    
    Evaluates ALL predicted neighbors, including those from unmatched cubicles:
    - Neighbors of matched cubicles are evaluated against GT
    - Neighbors of unmatched cubicles are counted as false positives
    - Missing GT cubicles contribute to false negatives for their neighbor relationships
    
    Args:
        parsed_response: Model output with internal_id-based neighbors
        ground_truth: Ground truth data
        name_threshold: Similarity threshold for fuzzy name matching
        
    Returns:
        Dictionary containing precision, recall, f1, and detailed metrics
    """
    # Build mapping from internal_id to GT
    internal_id_mapping = build_internal_id_mapping(
        parsed_response, ground_truth, name_threshold
    )
    
    # Separate matched vs unmatched cubicles
    matched_mappings = {
        iid: info for iid, info in internal_id_mapping.items()
        if info["match_type"] in ["id", "name"]
    }
    unmatched_mappings = {
        iid: info for iid, info in internal_id_mapping.items()
        if info["match_type"] == "unmatched"
    }
    
    # Get ground truth neighbors indexed by cubicle name
    gt_neighbors = {}
    for neighbor_entry in ground_truth.get("neighbors", []):
        name = neighbor_entry["name"]
        neighbors = set(neighbor_entry.get("neighbors", []))
        gt_neighbors[name] = neighbors
    
    # Process ALL predicted neighbors
    tp = 0
    fp = 0
    fn = 0
    
    # Track which GT cubicles had predictions evaluated
    gt_cubicles_with_predictions = set()
    
    for neighbor_entry in parsed_response.get("neighbors", []):
        source_internal_id = neighbor_entry["internal_id"]
        neighbor_internal_ids = neighbor_entry["neighbors"]
        
        # Check if source cubicle exists in our mapping
        if source_internal_id not in internal_id_mapping:
            # Invalid internal_id reference - count all neighbors as FP
            fp += len(neighbor_internal_ids)
            continue
        
        source_info = internal_id_mapping[source_internal_id]
        
        # If source cubicle is unmatched (hallucinated), all its neighbors are FP
        if source_info["match_type"] == "unmatched":
            fp += len(neighbor_internal_ids)
            continue
        
        # Source cubicle is matched - evaluate its neighbors against GT
        source_gt = source_info["gt_cubicle"]
        source_name = source_gt["name"]
        gt_neighbor_set = gt_neighbors.get(source_name, set())
        gt_cubicles_with_predictions.add(source_name)
        
        # Translate predicted neighbors to GT names, counting FPs for invalid/unmatched neighbors
        pred_neighbor_set = set()
        for neighbor_id in neighbor_internal_ids:
            if neighbor_id not in internal_id_mapping:
                # Invalid neighbor reference - count as FP
                fp += 1
                continue
            
            neighbor_info = internal_id_mapping[neighbor_id]
            if neighbor_info["match_type"] == "unmatched":
                # Neighbor is unmatched/hallucinated - count as FP
                fp += 1
            else:
                # Neighbor is matched - add to prediction set for comparison with GT
                neighbor_gt = neighbor_info["gt_cubicle"]
                pred_neighbor_set.add(neighbor_gt["name"])
        
        # Count matches using fuzzy name matching
        matched_pred = set()
        for gt_neighbor in gt_neighbor_set:
            matched = False
            for pred_neighbor in pred_neighbor_set:
                if pred_neighbor not in matched_pred:
                    similarity = _name_sim(gt_neighbor, pred_neighbor)
                    if similarity >= name_threshold:
                        tp += 1
                        matched_pred.add(pred_neighbor)
                        matched = True
                        break
            if not matched:
                fn += 1
        
        # Unmatched predictions are FPs
        fp += len(pred_neighbor_set - matched_pred)
    
    # Add FN for GT cubicles that were matched but had no neighbor predictions
    all_matched_gt_names = set()
    for info in matched_mappings.values():
        all_matched_gt_names.add(info["gt_cubicle"]["name"])
    
    for gt_name in all_matched_gt_names:
        if gt_name not in gt_cubicles_with_predictions:
            # This GT cubicle was identified but no neighbors were predicted
            # All its GT neighbors are false negatives
            gt_neighbor_set = gt_neighbors.get(gt_name, set())
            fn += len(gt_neighbor_set)
    
    # Add FN for GT cubicles that were completely missed (not in matched_mappings)
    gt_cubicle_names = set(gt_neighbors.keys())
    missed_gt_cubicles = gt_cubicle_names - all_matched_gt_names
    for gt_name in missed_gt_cubicles:
        # This GT cubicle was not identified at all
        # All its GT neighbors are false negatives
        gt_neighbor_set = gt_neighbors.get(gt_name, set())
        fn += len(gt_neighbor_set)
    
    # Compute precision, recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Track unmatched cubicles that have neighbor predictions (for debugging)
    unmatched_with_neighbors = []
    for neighbor_entry in parsed_response.get("neighbors", []):
        iid = neighbor_entry["internal_id"]
        if iid in unmatched_mappings:
            cubicle_info = unmatched_mappings[iid]["pred_cubicle"]
            unmatched_with_neighbors.append({
                "internal_id": iid,
                "name": cubicle_info.get("name"),
                "id": cubicle_info.get("id"),
                "neighbor_count": len(neighbor_entry["neighbors"]),
            })
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "matched_cubicles_evaluated": len(matched_mappings),
        "total_cubicles_in_gt": len(ground_truth.get("cubicles", [])),
        "total_cubicles_in_prediction": len(internal_id_mapping),
        "unmatched_cubicles": len(unmatched_mappings),
        "unmatched_cubicles_with_neighbors": unmatched_with_neighbors,
    }


def compute_cubicle_prf(
    gt_list: List[Dict], 
    pred_list: List[Dict], 
    name_threshold: float = 0.85
) -> Dict:
    """
    Compute precision, recall, f1 for cubicle pairs.
    Matching rules (in order):
      1) If pred.id == gt.id and name similarity >= name_threshold -> TP
      2) Else if pred.name exact or fuzzy matches a gt.name (>= name_threshold) -> TP (id mismatch tolerated)
      3) Otherwise, it's FP.
      
    Args:
        gt_list: Ground truth cubicle list
        pred_list: Predicted cubicle list
        name_threshold: Similarity threshold for fuzzy name matching
        
    Returns:
        Dictionary with precision, recall, f1, tp, fp, fn, and lists of unmatched preds & gts.
    """
    # copy ground truth lists
    gt_used = [False] * len(gt_list)
    tp = 0
    matched_pairs = []

    # try to match each predicted pair to a GT entry
    for pred in pred_list:
        p_id = pred.get("id", "").strip()
        p_name = pred.get("name", "").strip()
        best_gt_idx = None
        best_score = 0.0

        # first attempt: id match + name fuzzy
        for j, gt in enumerate(gt_list):
            if gt_used[j]:
                continue
            g_id = gt.get("id", "").strip()
            g_name = gt.get("name", "").strip()
            if g_id and p_id and g_id == p_id:
                score = _name_sim(p_name, g_name)
                if score >= name_threshold:
                    best_gt_idx = j
                    best_score = score
                    break  # strong match
                else:
                    # keep as candidate but continue searching for better
                    if score > best_score:
                        best_gt_idx = j
                        best_score = score

        # TODO: decide if we want this second step
        # second attempt: name-only fuzzy match if no id-match found
        # if best_gt_idx is None:
        #     for j, gt in enumerate(gt_list):
        #         if gt_used[j]:
        #             continue
        #         g_name = gt.get("name", "").strip()
        #         score = _name_sim(p_name, g_name)
        #         if score > best_score:
        #             best_gt_idx = j
        #             best_score = score

        # decide if it's a true positive
        if best_gt_idx is not None and best_score >= name_threshold:
            tp += 1
            gt_used[best_gt_idx] = True
            matched_pairs.append({"pred": pred, "gt": gt_list[best_gt_idx], "score": best_score})
        else:
            # remains unmatched -> will be counted as FP below
            pass

    fp = len(pred_list) - tp
    fn = sum(1 for used in gt_used if not used)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # prepare lists for debugging
    unmatched_preds = [p for p in pred_list if not any(p is m["pred"] for m in matched_pairs)]
    unmatched_gts = [gt_list[i] for i, used in enumerate(gt_used) if not used]

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "matched": matched_pairs,
        "unmatched_preds": unmatched_preds,
        "unmatched_gts": unmatched_gts,
    }


def compute_episode_metrics(
    parsed_response: Dict,
    gt: Dict,
    name_threshold: float = 0.85
) -> Dict:
    """
    Compute all metrics for a single episode.
    
    Args:
        parsed_response: Parsed response containing count, cubicles, and neighbors
        gt: Ground truth data
        name_threshold: Similarity threshold for fuzzy name matching
        
    Returns:
        Dictionary containing all computed metrics
    """
    metrics = {}
    
    # Count MAPE
    gt_count = int(gt.get("count", len(gt.get("cubicles", []))))
    pred_count = int(parsed_response["count"])
    metrics["count_mape_percent"] = mape_count(gt_count, pred_count)
    
    # Cubicle metrics
    gt_cubs = gt.get("cubicles", [])
    pred_cubs = parsed_response.get("cubicles", [])
    cub_metrics = compute_cubicle_prf(gt_cubs, pred_cubs, name_threshold=name_threshold)
    metrics["cubicles"] = {
        "precision": cub_metrics["precision"],
        "recall": cub_metrics["recall"],
        "f1": cub_metrics["f1"],
        "tp": cub_metrics["tp"],
        "fp": cub_metrics["fp"],
        "fn": cub_metrics["fn"],
        # include unmatched lists for debugging
        "unmatched_gt": cub_metrics["unmatched_gts"],
        "unmatched_pred": cub_metrics["unmatched_preds"],
    }
    
    # Neighbor metrics if GT has neighbor data - use internal_id mapping
    gt_neighbors = gt.get("neighbors", [])
    if gt_neighbors:
        neighbor_metrics = compute_neighbor_metrics(parsed_response, gt, name_threshold=name_threshold)
        metrics["neighbors"] = {
            "precision": neighbor_metrics["precision"],
            "recall": neighbor_metrics["recall"],
            "f1": neighbor_metrics["f1"],
            "tp": neighbor_metrics["tp"],
            "fp": neighbor_metrics["fp"],
            "fn": neighbor_metrics["fn"],
            "matched_cubicles_evaluated": neighbor_metrics["matched_cubicles_evaluated"],
            "total_cubicles_in_gt": neighbor_metrics["total_cubicles_in_gt"],
            "total_cubicles_in_prediction": neighbor_metrics["total_cubicles_in_prediction"],
            "unmatched_cubicles": neighbor_metrics["unmatched_cubicles"],
            "unmatched_cubicles_with_neighbors": neighbor_metrics["unmatched_cubicles_with_neighbors"],
        }
    
    return metrics


def aggregate_metrics(metrics_list: List[Dict]) -> Dict:
    """
    Aggregate metrics across multiple episodes.
    
    Args:
        metrics_list: List of per-episode metrics
        
    Returns:
        Dictionary containing aggregated metrics
    """
    if not metrics_list:
        return {}
    
    # aggregate count MAPE
    count_mape_vals = [m["count_mape_percent"] for m in metrics_list if "count_mape_percent" in m]
    avg_count_mape = sum(count_mape_vals) / len(count_mape_vals) if count_mape_vals else None

    # aggregate cubicle PRF (macro average across episodes that reported them)
    cub_entries = [m["cubicles"] for m in metrics_list if "cubicles" in m]
    precision_vals = [c["precision"] for c in cub_entries if "precision" in c]
    recall_vals = [c["recall"] for c in cub_entries if "recall" in c]
    f1_vals = [c["f1"] for c in cub_entries if "f1" in c]

    avg_precision = sum(precision_vals) / len(precision_vals) if precision_vals else None
    avg_recall = sum(recall_vals) / len(recall_vals) if recall_vals else None
    avg_f1 = sum(f1_vals) / len(f1_vals) if f1_vals else None

    # aggregate counts
    total_tp = sum(c.get("tp", 0) for c in cub_entries)
    total_fp = sum(c.get("fp", 0) for c in cub_entries)
    total_fn = sum(c.get("fn", 0) for c in cub_entries)

    # aggregate neighbor metrics if available
    neighbor_entries = [m["neighbors"] for m in metrics_list if "neighbors" in m]
    neighbor_precision_vals = [n["precision"] for n in neighbor_entries if "precision" in n]
    neighbor_recall_vals = [n["recall"] for n in neighbor_entries if "recall" in n]
    neighbor_f1_vals = [n["f1"] for n in neighbor_entries if "f1" in n]

    avg_neighbor_precision = sum(neighbor_precision_vals) / len(neighbor_precision_vals) if neighbor_precision_vals else None
    avg_neighbor_recall = sum(neighbor_recall_vals) / len(neighbor_recall_vals) if neighbor_recall_vals else None
    avg_neighbor_f1 = sum(neighbor_f1_vals) / len(neighbor_f1_vals) if neighbor_f1_vals else None

    total_neighbor_tp = sum(n.get("tp", 0) for n in neighbor_entries)
    total_neighbor_fp = sum(n.get("fp", 0) for n in neighbor_entries)
    total_neighbor_fn = sum(n.get("fn", 0) for n in neighbor_entries)

    final_metrics = {
        "episodes_aggregated": len(metrics_list),
        "count_mape": {
            "values": count_mape_vals,
            "average_percent": avg_count_mape,
        },
        "cubicles": {
            "per_episode_precision": precision_vals,
            "per_episode_recall": recall_vals,
            "per_episode_f1": f1_vals,
            "average_precision": avg_precision,
            "average_recall": avg_recall,
            "average_f1": avg_f1,
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn,
        },
    }

    # Add neighbor metrics to final output if available
    if neighbor_entries:
        final_metrics["neighbors"] = {
            "per_episode_precision": neighbor_precision_vals,
            "per_episode_recall": neighbor_recall_vals,
            "per_episode_f1": neighbor_f1_vals,
            "average_precision": avg_neighbor_precision,
            "average_recall": avg_neighbor_recall,
            "average_f1": avg_neighbor_f1,
            "total_tp": total_neighbor_tp,
            "total_fp": total_neighbor_fp,
            "total_fn": total_neighbor_fn,
        }

    return final_metrics


def load_metrics_from_dir(out_dir: str, num_episodes: int, split: bool = False) -> List[Dict]:
    """
    Load per-episode metrics from saved JSON files.
    
    Args:
        out_dir: Directory containing saved metrics files
        num_episodes: Number of episodes to load
        split: Whether to use split video naming convention
        
    Returns:
        List of metrics dictionaries
    """
    metrics_list = []
    
    for i in range(num_episodes):
        if split:
            metrics_path = os.path.join(out_dir, f"episode_0_720p_10fps_part_{i}_metrics.json")
        else:
            metrics_path = os.path.join(out_dir, f"episode_{i}_metrics.json")
        
        if os.path.exists(metrics_path):
            try:
                m = json.load(open(metrics_path, "r"))
                m["episode"] = i
                metrics_list.append(m)
            except Exception as e:
                print(f"Warning: Failed to load {metrics_path}: {e}")
    
    return metrics_list


def save_json(path: str, data: Dict):
    """
    Save data to JSON file.
    
    Args:
        path: File path to save to
        data: Data to save
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print("Saved", path)
