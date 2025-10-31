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


def compute_neighbor_metrics(
    gt_neighbors: List[Dict], 
    pred_neighbors: List[Dict], 
    name_threshold: float = 0.85
) -> Dict:
    """
    Compute metrics for neighbor relationships.
    For each person, we compare their neighbor list to the ground truth.
    
    Args:
        gt_neighbors: Ground truth neighbor data
        pred_neighbors: Predicted neighbor data
        name_threshold: Similarity threshold for fuzzy name matching
        
    Returns:
        Dictionary containing precision, recall, f1, and per-person metrics
    """
    # Build dictionaries for easier lookup
    gt_dict = {item["name"]: set(item["neighbors"]) for item in gt_neighbors}
    pred_dict = {item["name"]: set(item["neighbors"]) for item in pred_neighbors}
    
    # Track metrics
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    per_person_metrics = []
    
    # Get all unique names from both GT and predictions
    all_names = set(gt_dict.keys()) | set(pred_dict.keys())
    
    for name in all_names:
        gt_neighbors_set = gt_dict.get(name, set())
        pred_neighbors_set = pred_dict.get(name, set())
        
        # Use fuzzy matching for neighbor names
        matched_pred = set()
        tp_local = 0
        
        for gt_neighbor in gt_neighbors_set:
            matched = False
            for pred_neighbor in pred_neighbors_set:
                if pred_neighbor not in matched_pred:
                    similarity = _name_sim(gt_neighbor, pred_neighbor)
                    if similarity >= name_threshold:
                        tp_local += 1
                        matched_pred.add(pred_neighbor)
                        matched = True
                        break
            
            if not matched:
                # This GT neighbor was not found in predictions
                pass
        
        fp_local = len(pred_neighbors_set) - tp_local
        fn_local = len(gt_neighbors_set) - tp_local
        
        total_tp += tp_local
        total_fp += fp_local
        total_fn += fn_local
        
        # Per-person metrics for debugging
        person_precision = tp_local / (tp_local + fp_local) if (tp_local + fp_local) > 0 else 0.0
        person_recall = tp_local / (tp_local + fn_local) if (tp_local + fn_local) > 0 else 0.0
        person_f1 = 2 * person_precision * person_recall / (person_precision + person_recall) if (person_precision + person_recall) > 0 else 0.0
        
        per_person_metrics.append({
            "name": name,
            "tp": tp_local,
            "fp": fp_local,
            "fn": fn_local,
            "precision": person_precision,
            "recall": person_recall,
            "f1": person_f1,
            "gt_neighbors": list(gt_neighbors_set),
            "pred_neighbors": list(pred_neighbors_set),
        })
    
    # Overall metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "per_person": per_person_metrics,
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
    
    # Neighbor metrics if GT has neighbor data
    gt_neighbors = gt.get("neighbors", [])
    if gt_neighbors:
        pred_neighbors = parsed_response.get("neighbors", [])
        neighbor_metrics = compute_neighbor_metrics(gt_neighbors, pred_neighbors, name_threshold=name_threshold)
        metrics["neighbors"] = {
            "precision": neighbor_metrics["precision"],
            "recall": neighbor_metrics["recall"],
            "f1": neighbor_metrics["f1"],
            "tp": neighbor_metrics["tp"],
            "fp": neighbor_metrics["fp"],
            "fn": neighbor_metrics["fn"],
            # include per-person metrics for debugging
            "per_person": neighbor_metrics["per_person"],
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
