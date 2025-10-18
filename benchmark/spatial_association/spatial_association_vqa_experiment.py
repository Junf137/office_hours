import os
import time
import json
import difflib
import argparse
from typing import List, Tuple, Dict
from google import genai
from pydantic import BaseModel
from dotenv import load_dotenv

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
GROUND_TRUTH_DIR = "benchmark/spatial_association/ground_truth"  # files: episode_{i}_gt.json
OUT_DIR = "output" 

# --- Allowed relations ---
ALLOWED_REL = {"left_of", "right_of"}

# --- Pydantic schemas to structure VLM output ---
class CubiclePair(BaseModel):
    id: str
    name: str

class SimpleEdge(BaseModel):
    source: str   # owner name string (e.g., "Amy")
    target: str   # owner name string (e.g., "Jason")
    relation: str # "left_of" or "right_of"

class CombinedResponse(BaseModel):
    count: int                # number of cubicles with readable names
    cubicles: List[CubiclePair]  # list of id<->name pairs (only include cubicles that have readable names)
    edges: List[SimpleEdge]      # left_of / right_of relations between names

# ---------------------------
# Prompt
# ---------------------------
# TODO: Decide if to keep (yes for now): "count" must equal the number of entries in "cubicles"
# TODO: Add some hinting for the direction (e.g., left_of means source is to the left of target from camera POV)
# TODO: Adjacency (ie just neighbors) or all pairs?
# TODO: Allow left_of, right_of and across from
# TODO: Try with different resolution and framerate videos
PROMPT = """
You are given a video surveying an office with multiple cubicles.
Produce ONE strict JSON object (no commentary) with exactly three keys: "count", "cubicles", and "edges".

Schema:
{
  "count": <integer>,
  "cubicles": [ {"id":"2008M","name":"Amy"}, ... ],
  "edges": [ {"source":"Amy","target":"Jason","relation":"left_of"}, ... ]
}

Rules:
- Include ONLY cubicles where an owner name appears in the video.
- Each cubicle entry must contain both an "id" and "name":
    - Use the visible cubicle ID if the cubicle has one assigned (e.g., "2008M").
    - If a cubicle does not have any ID shown or assigned in the video, set "id": "N/A".
- "count" must equal the number of entries in "cubicles".
- "cubicles" must include ONLY cubicles for which a readable owner name appears in the video. Each entry must contain an id (visible id like 2008 M) and the exact visible name string.
- "edges" must reference the owner names exactly as they appear in "cubicles".
- Use ONLY the relations "left_of" or "right_of".
"""

# ---------------------------
# Metric helpers
# ---------------------------
def load_gt_for_episode(i: int) -> Dict:
    path = os.path.join(GROUND_TRUTH_DIR, f"episode_{i}_gt.json")
    if not os.path.exists(path):
        return {}
    return json.load(open(path, "r"))

def mape_count(gt_count: int, pred_count: int) -> float:
    """
    Mean Absolute Percentage Error for single scalar counts.
    """
    # handle zero gt case
    if gt_count == 0:
        return 0.0 if pred_count == 0 else 100.0
    # compute |(pred-gt)/gt| * 100
    return abs(pred_count - gt_count) / gt_count * 100.0

def _name_sim(a: str, b: str) -> float:
    if a is None or b is None:
        return 0.0
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()

def compute_cubicle_prf(gt_list: List[Dict], pred_list: List[Dict], name_threshold: float = 0.85) -> Dict:
    """
    Compute precision, recall, f1 for cubicle pairs.
    Matching rules (in order):
      1) If pred.id == gt.id and name similarity >= name_threshold -> TP
      2) Else if pred.name exact or fuzzy matches a gt.name (>= name_threshold) -> TP (id mismatch tolerated)
      3) Otherwise, it's FP.
    Returns dict with precision, recall, f1, tp, fp, fn, and lists of unmatched preds & gts.
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

# ---------------------------
# I/O helper
# ---------------------------
def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print("Saved", path)

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

        bad_rels = [e for e in parsed.edges if e.relation not in ALLOWED_REL]
        if bad_rels:
            print("Warning: unsupported relations found:", {e.relation for e in bad_rels})

        # check edge references exist in cubicle names
        cubicle_names = {c.name for c in parsed.cubicles}
        missing_refs = [e for e in parsed.edges if e.source not in cubicle_names or e.target not in cubicle_names]
        if missing_refs:
            print("Warning: some edges reference names not present in cubicles list:")
            for e in missing_refs:
                print(f"  {e.source} -> {e.target} ({e.relation})")

        # ---------------------------
        # Metrics (if GT available)
        # ---------------------------
        gt = load_gt_for_episode(i)
        metrics = {}
        if gt:
            gt_count = int(gt.get("count", len(gt.get("cubicles", []))))
            pred_count = int(parsed.count)
            metrics["count_mape_percent"] = mape_count(gt_count, pred_count)

            gt_cubs = gt.get("cubicles", [])
            pred_cubs = [c.model_dump() for c in parsed.cubicles]
            cub_metrics = compute_cubicle_prf(gt_cubs, pred_cubs, name_threshold=0.85)
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

            # save metrics
            metrics_path = os.path.join(out_dir, f"episode_{i}_metrics.json")
            save_json(metrics_path, metrics)
        else:
            print("No ground-truth file found; skipping metrics.")

        # save parsed result
        out = {
            "count": parsed.count,
            "cubicles": [c.model_dump() for c in parsed.cubicles],
            "edges": [e.model_dump() for e in parsed.edges],
        }
        save_path = os.path.join(out_dir, f"episode_{i}_combined.json")
        save_json(save_path, out)

        # concise summary
        print(f"Episode {i}: named cubicles = {parsed.count}; edges = {len(parsed.edges)}")
        if gt:
            print("Metrics (summary):", {k: metrics[k] for k in metrics})
        for c in parsed.cubicles:
            print(f"  {c.id} : {c.name}")
        for e in parsed.edges:
            print(f"  {e.source} -> {e.target} : {e.relation}")
        print("-" * 60)

    # ---------------------------
    # Final aggregation across episodes
    # ---------------------------
    metrics_list = globals().get("metrics_list", None)
    # If a previous run didn't initialize an accumulator inside the loop, try to build one from saved files
    if metrics_list is None:
        metrics_list = []

    # Attempt to load per-episode metric files in out_dir as a fallback
    for i in range(num_episodes):
        metrics_path = os.path.join(out_dir, f"episode_{i}_metrics.json")
        if os.path.exists(metrics_path):
            try:
                m = json.load(open(metrics_path, "r"))
                m["episode"] = i
                # avoid duplicates
                if not any(existing.get("episode") == i for existing in metrics_list):
                    metrics_list.append(m)
            except Exception:
                pass

    if not metrics_list:
        print("No per-episode metrics found; final aggregation skipped.")
    else:
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

        final_path = os.path.join(out_dir, "final_metrics.json")
        save_json(final_path, final_metrics)
        print("Final aggregated metrics saved to", final_path)
        print("Summary:", {
            "episodes": final_metrics["episodes_aggregated"],
            "avg_count_mape_percent": final_metrics["count_mape"]["average_percent"],
            "avg_precision": final_metrics["cubicles"]["average_precision"],
            "avg_recall": final_metrics["cubicles"]["average_recall"],
            "avg_f1": final_metrics["cubicles"]["average_f1"],
        })