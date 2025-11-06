"""
Inference script for spatial association VQA experiment.
Generates raw model outputs and parsed JSON responses from video inputs.

python benchmark/spatial_association/run_inference.py -m gemini -o output/gemini
"""
import os
import json
import argparse
from typing import List, Dict
from pydantic import BaseModel
from dotenv import load_dotenv
from model_interface import create_model

load_dotenv()

# --- Constants ---
SPLIT = False  # whether to use split videos and ground truth
if SPLIT:
    NUM_EPISODES = 24
else:
    NUM_EPISODES = 6

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
# Helper functions
# ---------------------------
def save_json(path: str, data: Dict):
    """Save data to JSON file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {path}")


def run_inference(
    num_episodes: int,
    out_dir: str,
    model_provider: str = "gemini",
    model_config: Dict = None,
    split: bool = False,
    video_dir: str = None,
    prompt: str = None,
    response_schema = None,
) -> int:
    """
    Run inference on videos and save outputs.
    
    Args:
        num_episodes: Number of episodes to process
        out_dir: Output directory for results
        model_provider: Model provider name ("gemini" or "gpt4o")
        model_config: Model configuration dict (optional, uses defaults if None)
        split: Whether to use split videos
        video_dir: Custom video directory (optional)
        prompt: Custom prompt (optional, uses default if None)
        response_schema: Pydantic schema for response (optional, uses CombinedResponse if None)
        
    Returns:
        Number of episodes successfully processed
    """
    os.makedirs(out_dir, exist_ok=True)

    # Use default config if not provided
    if model_config is None:
        model_config = MODEL_CONFIGS.get(model_provider, MODEL_CONFIGS[MODEL_PROVIDER])
    
    # Use default prompt if not provided
    if prompt is None:
        prompt = PROMPT
    
    # Use default response schema if not provided
    if response_schema is None:
        response_schema = CombinedResponse

    # Initialize the model
    model = create_model(model_provider, **model_config)
    print(f"Using model: {model.get_model_name()}")
    print(f"Output directory: {out_dir}")
    print(f"Processing {num_episodes} episodes")
    print(f"Split mode: {split}")

    episodes_processed = 0

    for i in range(num_episodes):
        # Determine video path
        if video_dir:
            # Custom video directory provided
            if split:
                episode_path = os.path.join(video_dir, f"episode_0_720p_10fps_part_{i}.mp4")
            else:
                episode_path = os.path.join(video_dir, f"episode_{i}_720p_10fps.mp4")
        else:
            # Use default paths
            if split:
                episode_path = f"benchmark/spatial_association/split_videos/episode_0_720p_10fps_part_{i}.mp4"
            else:
                episode_path = f"data/global_changes_videos/episode_{i}_720p_10fps.mp4"
        
        if not os.path.exists(episode_path):
            print(f"Missing file: {episode_path}")
            continue

        # Generate response using the model interface
        print(f"\n{'='*60}")
        print(f"Processing episode {i}...")
        print(f"Video: {episode_path}")
        print(f"{'='*60}")
        
        try:
            resp_text = model.generate_response(episode_path, prompt, response_schema)
        except Exception as e:
            print(f"Error generating response for episode {i}: {e}")
            continue

        # Save raw response
        if split:
            raw_path = os.path.join(out_dir, f"episode_0_720p_10fps_part_{i}_raw.txt")
        else:
            raw_path = os.path.join(out_dir, f"episode_{i}_raw.txt")
        with open(raw_path, "w") as f:
            f.write(resp_text)
        print(f"Saved raw response to {raw_path}")

        # Validate & parse
        try:
            parsed = response_schema.model_validate_json(resp_text)
        except Exception as e:
            print(f"Validation failed: {e}")
            print(f"Raw response:\n{resp_text}")
            continue

        # Basic checks
        if hasattr(parsed, 'count') and hasattr(parsed, 'cubicles'):
            if parsed.count != len(parsed.cubicles):
                print(f"Warning: count ({parsed.count}) != number of cubicles returned ({len(parsed.cubicles)}).")

        # Save parsed result
        out = {
            "count": parsed.count,
            "cubicles": [c.model_dump() for c in parsed.cubicles],
            "neighbors": [n.model_dump() for n in parsed.neighbors],
        }
        if split:
            save_path = os.path.join(out_dir, f"episode_0_720p_10fps_part_{i}_combined.json")
        else:
            save_path = os.path.join(out_dir, f"episode_{i}_combined.json")
        save_json(save_path, out)
        
        episodes_processed += 1

    print("\n" + "="*60)
    print(f"Inference complete! Processed {episodes_processed}/{num_episodes} episodes")
    print(f"Results saved to: {out_dir}")
    print("="*60)
    
    return episodes_processed


# ---------------------------
# Main inference loop
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run spatial association VQA inference")
    parser.add_argument("-n", "--num-episodes", type=int, default=NUM_EPISODES,
                        help="Number of episodes to process (default: %(default)s)")
    parser.add_argument("-o", "--out-dir", type=str, default=OUT_DIR,
                        help="Output directory for results (default: %(default)s)")
    parser.add_argument("-m", "--model", type=str, default=MODEL_PROVIDER,
                        choices=["gemini", "gpt4o"],
                        help="Model provider to use (default: %(default)s)")
    parser.add_argument("--split", action="store_true",
                        help="Use split videos and ground truth")
    parser.add_argument("--video-dir", type=str, default=None,
                        help="Custom video directory (optional)")
    args = parser.parse_args()

    # Run inference using the extracted function
    run_inference(
        num_episodes=args.num_episodes,
        out_dir=args.out_dir,
        model_provider=args.model,
        model_config=MODEL_CONFIGS.get(args.model, MODEL_CONFIGS[MODEL_PROVIDER]),
        split=args.split or SPLIT,
        video_dir=args.video_dir,
        prompt=PROMPT,
        response_schema=CombinedResponse,
    )
