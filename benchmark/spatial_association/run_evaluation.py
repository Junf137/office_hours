"""
Evaluation script for spatial association VQA experiment.
Loads parsed model outputs and ground truth, computes evaluation metrics.

python benchmark/spatial_association/run_evaluation.py -i output/gemini -o output/gemini

"""
import os
import json
import argparse
from typing import List, Dict
from evaluation_utils import (
    load_gt_for_episode,
    compute_episode_metrics,
    aggregate_metrics,
    load_metrics_from_dir,
    save_json,
)

# --- Constants ---
SPLIT = False  # whether to use split videos and ground truth
if SPLIT:
    NUM_EPISODES = 24
    GROUND_TRUTH_DIR = "benchmark/spatial_association/ground_truth_split"  # files: episode_0_gt_part_{i}.json
else:
    NUM_EPISODES = 6
    GROUND_TRUTH_DIR = "benchmark/spatial_association/ground_truth"  # files: episode_{i}_gt.json

INFERENCE_DIR = "output/4_neigh_metrics"  # Directory containing inference outputs
OUT_DIR = "output/4_neigh_metrics"  # Directory for evaluation outputs


def run_evaluation(
    num_episodes: int,
    inference_dir: str,
    out_dir: str,
    ground_truth_dir: str,
    split: bool = False,
    name_threshold: float = 0.85,
    verbose: bool = True,
) -> Dict:
    """
    Run evaluation on inference outputs against ground truth.
    
    Args:
        num_episodes: Number of episodes to evaluate
        inference_dir: Directory containing inference outputs
        out_dir: Output directory for evaluation results
        ground_truth_dir: Ground truth directory
        split: Whether to use split videos
        name_threshold: Name similarity threshold for fuzzy matching
        verbose: Whether to print detailed progress
        
    Returns:
        Dictionary containing final aggregated metrics
    """
    os.makedirs(out_dir, exist_ok=True)

    if verbose:
        print(f"Inference directory: {inference_dir}")
        print(f"Ground truth directory: {ground_truth_dir}")
        print(f"Output directory: {out_dir}")
        print(f"Evaluating {num_episodes} episodes")
        print(f"Split mode: {split}")
        print(f"Name similarity threshold: {name_threshold}")
        print()

    episodes_evaluated = 0
    episodes_skipped = 0

    for i in range(num_episodes):
        # Load parsed inference output
        if split:
            inference_path = os.path.join(inference_dir, f"episode_0_720p_10fps_part_{i}_combined.json")
        else:
            inference_path = os.path.join(inference_dir, f"episode_{i}_combined.json")
        
        if not os.path.exists(inference_path):
            if verbose:
                print(f"Skipping episode {i}: inference output not found at {inference_path}")
            episodes_skipped += 1
            continue

        try:
            with open(inference_path, "r") as f:
                parsed_response = json.load(f)
        except Exception as e:
            if verbose:
                print(f"Skipping episode {i}: failed to load inference output: {e}")
            episodes_skipped += 1
            continue

        # Load ground truth
        gt = load_gt_for_episode(i, ground_truth_dir, split)
        if not gt:
            if verbose:
                print(f"Skipping episode {i}: no ground truth file found")
            episodes_skipped += 1
            continue

        # Compute metrics
        if verbose:
            print(f"Evaluating episode {i}...")
        metrics = compute_episode_metrics(parsed_response, gt, name_threshold=name_threshold)

        # Save metrics
        if split:
            metrics_path = os.path.join(out_dir, f"episode_0_720p_10fps_part_{i}_metrics.json")
        else:
            metrics_path = os.path.join(out_dir, f"episode_{i}_metrics.json")
        save_json(metrics_path, metrics)
        
        episodes_evaluated += 1

    if verbose:
        print()
        print("="*60)
        print(f"Evaluated {episodes_evaluated} episodes")
        print(f"Skipped {episodes_skipped} episodes")
        print("="*60)
        print()

    # ---------------------------
    # Final aggregation across episodes
    # ---------------------------
    if episodes_evaluated == 0:
        if verbose:
            print("No episodes evaluated; final aggregation skipped.")
        return {}
    
    if verbose:
        print("Aggregating metrics across episodes...")
    
    # Load all per-episode metrics from saved files
    metrics_list = load_metrics_from_dir(out_dir, num_episodes, split)

    if not metrics_list:
        if verbose:
            print("No per-episode metrics found; final aggregation skipped.")
        return {}
    
    # Aggregate all metrics across episodes
    final_metrics = aggregate_metrics(metrics_list)

    # Save aggregated metrics
    final_path = os.path.join(out_dir, "final_metrics.json")
    save_json(final_path, final_metrics)
    
    # Print summary
    if verbose:
        print()
        print("="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Episodes aggregated: {final_metrics['episodes_aggregated']}")
        print(f"Average count MAPE: {final_metrics['count_mape']['average_percent']:.2f}%")
        print()
        print("Cubicles:")
        print(f"  Average Precision: {final_metrics['cubicles']['average_precision']:.4f}")
        print(f"  Average Recall:    {final_metrics['cubicles']['average_recall']:.4f}")
        print(f"  Average F1:        {final_metrics['cubicles']['average_f1']:.4f}")
        print(f"  Total TP/FP/FN:    {final_metrics['cubicles']['total_tp']}/{final_metrics['cubicles']['total_fp']}/{final_metrics['cubicles']['total_fn']}")
        
        if "neighbors" in final_metrics:
            print()
            print("Neighbors:")
            print(f"  Average Precision: {final_metrics['neighbors']['average_precision']:.4f}")
            print(f"  Average Recall:    {final_metrics['neighbors']['average_recall']:.4f}")
            print(f"  Average F1:        {final_metrics['neighbors']['average_f1']:.4f}")
            print(f"  Total TP/FP/FN:    {final_metrics['neighbors']['total_tp']}/{final_metrics['neighbors']['total_fp']}/{final_metrics['neighbors']['total_fn']}")
        
        print("="*60)
    
    return final_metrics


# ---------------------------
# Main evaluation loop
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run spatial association VQA evaluation")
    parser.add_argument("-n", "--num-episodes", type=int, default=NUM_EPISODES,
                        help="Number of episodes to evaluate (default: %(default)s)")
    parser.add_argument("-i", "--inference-dir", type=str, default=INFERENCE_DIR,
                        help="Directory containing inference outputs (default: %(default)s)")
    parser.add_argument("-o", "--out-dir", type=str, default=OUT_DIR,
                        help="Output directory for evaluation results (default: %(default)s)")
    parser.add_argument("-g", "--ground-truth-dir", type=str, default=GROUND_TRUTH_DIR,
                        help="Ground truth directory (default: %(default)s)")
    parser.add_argument("--split", action="store_true",
                        help="Use split videos and ground truth")
    parser.add_argument("--name-threshold", type=float, default=0.85,
                        help="Name similarity threshold for fuzzy matching (default: 0.85)")
    args = parser.parse_args()

    # Run evaluation using the extracted function
    run_evaluation(
        num_episodes=args.num_episodes,
        inference_dir=args.inference_dir,
        out_dir=args.out_dir,
        ground_truth_dir=args.ground_truth_dir,
        split=args.split or SPLIT,
        name_threshold=args.name_threshold,
        verbose=True,
    )
