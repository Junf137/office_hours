import os
import json
import argparse
from typing import List, Dict
from pydantic import BaseModel
from dotenv import load_dotenv
from evaluation_utils import (
    load_gt_for_episode,
    compute_episode_metrics,
    aggregate_metrics,
    load_metrics_from_dir,
    save_json,
)
from model_interface import create_model

load_dotenv()

# --- Constants ---
SPLIT = False  # whether to use split videos and ground truth
if SPLIT:
    NUM_EPISODES = 24
    GROUND_TRUTH_DIR = "benchmark/spatial_association/ground_truth_split"  # files: episode_0_gt_part_{i}.json
else:
    NUM_EPISODES = 6
    GROUND_TRUTH_DIR = "benchmark/spatial_association/ground_truth"  # files: episode_{i}_gt.json

OUT_DIR = "output/4_neigh_metrics" 
os.makedirs(OUT_DIR, exist_ok=True)

# --- Model Configuration ---
# Supported models: "gemini", "gpt4o"
MODEL_PROVIDER = "gemini"

# Provider-specific model names
MODEL_CONFIGS = {
    "gemini": {
        "model_name": "gemini-2.5-pro",
        "temperature": 0.0,
    },
    "gpt4o": {
        "model_name": "gpt-4o",
        "temperature": 0.0,
        "num_frames": 10,  # Use -1 for all frames, or specify a number (it will be uniformly sampled)
    }
}

# --- Pydantic schemas to structure VLM output ---
class CubiclePair(BaseModel):
    id: str
    name: str

class NeighborItem(BaseModel):
    name: str                # owner name on this cubicle
    neighbors: List[str]     # list of neighbor owner names (strings)

class CombinedResponse(BaseModel):
    count: int                # number of cubicles with readable names
    cubicles: List[CubiclePair]  # list of id<->name pairs (only include cubicles that have readable names)
    neighbors: List[NeighborItem]       # neighbor lists per cubicle (names)

# ---------------------------
# Prompt
# ---------------------------
PROMPT = """
You are given a video surveying an office with multiple cubicles.
Produce ONE strict JSON object (no commentary) with exactly three keys: "count", "cubicles", and "neighbors".

Schema:
{
  "count": <integer>,
  "cubicles": [ {"id":"2008M","name":"Amy"}, ... ],
  "neighbors": [ {"name":"Amy","neighbors":["Jason","Lauren"]}, ... ]
}

Rules:
- Include ONLY cubicles where an owner name appears in the video.
- Each cubicle entry must contain both an "id" and "name":
    - Use the visible cubicle ID if the cubicle has one assigned (e.g., "2008M").
    - If a cubicle does not have any ID shown or assigned in the video, set "id": "N/A".
- "count" must equal the number of entries in "cubicles".
- "cubicles" must include ONLY cubicles for which a readable owner name appears in the video. Each entry must contain an id (like 2008 M or N/A if not present) and the exact visible name string.
- "neighbors" must list direct neighboring cubicles by owner name (only include neighbors whose names appear in "cubicles").
    - Neighbor relationships must be bidirectional and consistent. If Jason is listed as a neighbor of Amy, then Amy MUST be listed as a neighbor of Jason. Ensure all neighbor relationships are symmetric.
"""

# ---------------------------
# Main loop: prompting & evaluation
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run spatial association VQA experiment")
    parser.add_argument("-n", "--num-episodes", type=int, default=NUM_EPISODES,
                        help="Number of episodes to process (default: %(default)s)")
    parser.add_argument("-o", "--out-dir", type=str, default=OUT_DIR,
                        help="Output directory for results (default: %(default)s)")
    parser.add_argument("-m", "--model", type=str, default=MODEL_PROVIDER,
                        choices=["gemini", "gpt4o"],
                        help="Model provider to use (default: %(default)s)")
    args = parser.parse_args()

    out_dir = args.out_dir
    num_episodes = args.num_episodes
    model_provider = args.model

    os.makedirs(out_dir, exist_ok=True)

    # Initialize the model
    model_config = MODEL_CONFIGS.get(model_provider, MODEL_CONFIGS[MODEL_PROVIDER])
    model = create_model(model_provider, **model_config)
    print(f"Using model: {model.get_model_name()}")

    for i in range(num_episodes):
        if SPLIT:
            episode_path = f"benchmark/spatial_association/split_videos/episode_0_720p_10fps_part_{i}.mp4"
        else:
            episode_path = f"data/global_changes_videos/episode_{i}_720p_10fps.mp4"
        if not os.path.exists(episode_path):
            print("Missing file:", episode_path)
            continue

        # Generate response using the model interface
        print(f"Processing episode {i}...")
        resp_text = model.generate_response(episode_path, PROMPT, CombinedResponse)

        # save raw response
        if SPLIT:
            raw_path = os.path.join(out_dir, f"episode_0_720p_10fps_part_{i}_raw.txt")
        else:
            raw_path = os.path.join(out_dir, f"episode_{i}_raw.txt")
        with open(raw_path, "w") as f:
            f.write(resp_text)
        print("Saved raw response to", raw_path)

        # validate & parse
        try:
            parsed = CombinedResponse.model_validate_json(resp_text)
        except Exception as e:
            print("Validation failed:", e)
            print("Raw response:\n", resp_text)
            continue

        # basic checks
        if parsed.count != len(parsed.cubicles):
            print(f"Warning: count ({parsed.count}) != number of cubicles returned ({len(parsed.cubicles)}).")

        # ---------------------------
        # Metrics (if GT available)
        # ---------------------------
        gt = load_gt_for_episode(i, GROUND_TRUTH_DIR, SPLIT)
        if gt:
            # Prepare parsed response in dict format
            parsed_response = {
                "count": parsed.count,
                "cubicles": [c.model_dump() for c in parsed.cubicles],
                "neighbors": [n.model_dump() for n in parsed.neighbors],
            }
            
            # Compute all metrics for this episode
            metrics = compute_episode_metrics(parsed_response, gt, name_threshold=0.85)

            # save metrics
            if SPLIT:
                metrics_path = os.path.join(out_dir, f"episode_0_720p_10fps_part_{i}_metrics.json")
            else:
                metrics_path = os.path.join(out_dir, f"episode_{i}_metrics.json")
            save_json(metrics_path, metrics)
        else:
            print("No ground-truth file found; skipping metrics.")

        # save parsed result
        out = {
            "count": parsed.count,
            "cubicles": [c.model_dump() for c in parsed.cubicles],
            "neighbors": [n.model_dump() for n in parsed.neighbors],
        }
        if SPLIT:
            save_path = os.path.join(out_dir, f"episode_0_720p_10fps_part_{i}_combined.json")
        else:
            save_path = os.path.join(out_dir, f"episode_{i}_combined.json")
        save_json(save_path, out)

    # ---------------------------
    # Final aggregation across episodes
    # ---------------------------
    # Load all per-episode metrics from saved files
    metrics_list = load_metrics_from_dir(out_dir, num_episodes, SPLIT)

    if not metrics_list:
        print("No per-episode metrics found; final aggregation skipped.")
    else:
        # Aggregate all metrics across episodes
        final_metrics = aggregate_metrics(metrics_list)

        # Save aggregated metrics
        final_path = os.path.join(out_dir, "final_metrics.json")
        save_json(final_path, final_metrics)
        print("Final aggregated metrics saved to", final_path)
        
        # Print summary
        summary = {
            "episodes": final_metrics["episodes_aggregated"],
            "avg_count_mape_percent": final_metrics["count_mape"]["average_percent"],
            "cubicles": {
                "avg_precision": final_metrics["cubicles"]["average_precision"],
                "avg_recall": final_metrics["cubicles"]["average_recall"],
                "avg_f1": final_metrics["cubicles"]["average_f1"],
            }
        }
        
        if "neighbors" in final_metrics:
            summary["neighbors"] = {
                "avg_precision": final_metrics["neighbors"]["average_precision"],
                "avg_recall": final_metrics["neighbors"]["average_recall"],
                "avg_f1": final_metrics["neighbors"]["average_f1"],
            }
        
        print("Summary:", summary)