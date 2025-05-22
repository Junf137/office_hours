# List of original .MOV file base names (without extension)
files=(
  "episode_0"
  "episode_1"
  "episode_2"
  "episode_3"
  "episode_4"
  "episode_5"
)

# Navigate to MASt3R-SLAM directory
cd ./MASt3R-SLAM

# Dataset base path and config file
base_path="../segmented_videos"
config_file="config/base.yaml"

for base in "${files[@]}"; do
  echo "Searching for chunks of $base..."

  # Find all matching chunked files for this base name
  for chunk in ${base_path}/${base}_5fps_part*.mp4; do
    # If no match, skip
    [ -e "$chunk" ] || continue

    echo "Processing $chunk..."
    python main.py --dataset "$chunk" --config "$config_file" --no-viz
    echo "Finished processing $chunk."
  done
done

echo "All parts processed."
