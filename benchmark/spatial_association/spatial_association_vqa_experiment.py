import os
import time
import json
import argparse
from typing import List, Dict
from google import genai
from pydantic import BaseModel
from dotenv import load_dotenv
from evaluation_utils import (
    load_gt_for_episode,
    compute_episode_metrics,
    aggregate_metrics,
    load_metrics_from_dir,
    save_json,
)

load_dotenv()

#  --- Gemini client setup ---
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
# Determine which model to use
# print("List of models that support generateContent:\n")
# for m in client.models.list():
#     for action in m.supported_actions:
#         if action == "generateContent":
#             print(m.name)

# GEMIMI_MODEL = "gemini-2.5-flash-lite"
# GEMIMI_MODEL = "gemini-2.5-flash"
GEMIMI_MODEL = "gemini-2.5-pro"

# --- Constants ---
NUM_EPISODES = 6
SPLIT = False  # whether to use split videos and ground truth
if SPLIT:
    NUM_EPISODES = 24
    GROUND_TRUTH_DIR = "benchmark/spatial_association/ground_truth_split"  # files: episode_{i}_gt.json
else:
    GROUND_TRUTH_DIR = "benchmark/spatial_association/ground_truth"  # files: episode_{i}_gt.json

OUT_DIR = "output/4_neigh_metrics" 
os.makedirs(OUT_DIR, exist_ok=True)

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
    args = parser.parse_args()

    out_dir = args.out_dir
    num_episodes = args.num_episodes

    os.makedirs(out_dir, exist_ok=True)

    for i in range(num_episodes):
        if SPLIT:
            episode_path = f"benchmark/spatial_association/split_videos/episode_0_720p_10fps_part_{i}.mp4"
        else:
            episode_path = f"data/global_changes_videos/episode_{i}_720p_10fps.mp4"
        if not os.path.exists(episode_path):
            print("Missing file:", episode_path)
            continue

        print("Uploading", episode_path)
        myfile = client.files.upload(file=episode_path)

        # Wait until the file is processed
        while not myfile.state or myfile.state.name != "ACTIVE":
            print("Waiting for file processing... state:", myfile.state)
            time.sleep(3)
            myfile = client.files.get(name=myfile.name)

        # end request prompt for the episode
        resp = client.models.generate_content(
            model=GEMIMI_MODEL,
            contents=[myfile, PROMPT],
            config={
                "temperature": 0.0,
                "response_mime_type": "application/json",
                "response_schema": CombinedResponse,
            },
        )

        # save raw response
        if SPLIT:
            raw_path = os.path.join(out_dir, f"episode_0_720p_10fps_part_{i}_raw.txt")
        else:
            raw_path = os.path.join(out_dir, f"episode_{i}_raw.txt")
        with open(raw_path, "w") as f:
            f.write(resp.text)
        print("Saved raw response to", raw_path)

        # validate & parse
        try:
            parsed = CombinedResponse.model_validate_json(resp.text)
        except Exception as e:
            print("Validation failed:", e)
            print("Raw response:\n", resp.text)
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