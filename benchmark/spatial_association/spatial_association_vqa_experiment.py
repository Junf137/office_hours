import os
import argparse
from dotenv import load_dotenv
from run_inference import run_inference, MODEL_CONFIGS, PROMPT, CombinedResponse
from run_evaluation import run_evaluation

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
MODEL_PROVIDER = "gemini"

# ---------------------------
# Main pipeline: inference + evaluation
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run complete spatial association VQA experiment (inference + evaluation)"
    )
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
    parser.add_argument("-g", "--ground-truth-dir", type=str, default=GROUND_TRUTH_DIR,
                        help="Ground truth directory (default: %(default)s)")
    parser.add_argument("--name-threshold", type=float, default=0.85,
                        help="Name similarity threshold for fuzzy matching (default: 0.85)")
    parser.add_argument("--skip-inference", action="store_true",
                        help="Skip inference and only run evaluation on existing outputs")
    parser.add_argument("--skip-evaluation", action="store_true",
                        help="Skip evaluation and only run inference")
    args = parser.parse_args()

    split = args.split or SPLIT
    
    print("="*60)
    print("SPATIAL ASSOCIATION VQA FULL PIPELINE")
    print("="*60)
    print(f"Mode: {'Split' if split else 'Full'} videos")
    print(f"Episodes: {args.num_episodes}")
    print(f"Output directory: {args.out_dir}")
    print(f"Model: {args.model}")
    if not args.skip_inference:
        print(f"Video directory: {args.video_dir or 'default'}")
    if not args.skip_evaluation:
        print(f"Ground truth directory: {args.ground_truth_dir}")
        print(f"Name threshold: {args.name_threshold}")
    print("="*60)
    print()

    # ---------------------------
    # Step 1: Run Inference
    # ---------------------------
    if not args.skip_inference:
        print("\n" + "="*60)
        print("STEP 1: RUNNING INFERENCE")
        print("="*60 + "\n")
        
        model_config = MODEL_CONFIGS.get(args.model, MODEL_CONFIGS[MODEL_PROVIDER])
        episodes_processed = run_inference(
            num_episodes=args.num_episodes,
            out_dir=args.out_dir,
            model_provider=args.model,
            model_config=model_config,
            split=split,
            video_dir=args.video_dir,
            prompt=PROMPT,
            response_schema=CombinedResponse,
        )
        
        if episodes_processed == 0:
            print("\nNo episodes were successfully processed during inference.")
            if not args.skip_evaluation:
                print("Skipping evaluation step.")
            exit(1)
    else:
        print("\nSkipping inference (--skip-inference flag set)")
    
    # ---------------------------
    # Step 2: Run Evaluation
    # ---------------------------
    if not args.skip_evaluation:
        print("\n" + "="*60)
        print("STEP 2: RUNNING EVALUATION")
        print("="*60 + "\n")
        
        final_metrics = run_evaluation(
            num_episodes=args.num_episodes,
            inference_dir=args.out_dir,
            out_dir=args.out_dir,
            ground_truth_dir=args.ground_truth_dir,
            split=split,
            name_threshold=args.name_threshold,
            verbose=True,
        )
        
        if not final_metrics:
            print("\nEvaluation produced no results.")
        else:
            print("\n" + "="*60)
            print("PIPELINE COMPLETE!")
            print("="*60)
            print(f"All results saved to: {args.out_dir}")
            print("="*60)
    else:
        print("\nSkipping evaluation (--skip-evaluation flag set)")
        print("\n" + "="*60)
        print("INFERENCE COMPLETE!")
        print("="*60)
        print(f"Results saved to: {args.out_dir}")
        print("="*60)