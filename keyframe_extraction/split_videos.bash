#!/bin/bash

# Directory containing the input MP4 files
input_dir="../data/global_changes_videos"

# Directory to store output videos
output_dir="segmented_videos"
mkdir -p "$output_dir"

# List of .mp4 files
files=(
  "episode_0.mp4"
  "episode_1.mp4"
  "episode_2.mp4"
  "episode_3.mp4"
  "episode_4.mp4"
  "episode_5.mp4"
)

# Duration of each segment (in seconds)
segment_length=300     # 300 seconds
overlap=30             # 30 seconds
step=$((segment_length - overlap))

for mp4file in "${files[@]}"; do
  # Full path to input file
  input_path="${input_dir}/${mp4file}"

  # Get the base name without extension
  base="${mp4file%.mp4}"

  echo "Processing $input_path..."

  # Output paths
  downsampled="${output_dir}/${base}_5fps.mp4"

  # Downsample FPS to 5
  ffmpeg -y -i "$input_path" -vf "fps=5" "$downsampled"

  # Get video duration in seconds
  duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$downsampled")
  duration=${duration%.*}  # Convert to integer

  echo "Duration: $duration seconds"

  # Loop and extract overlapping segments
  start=0
  count=0
  while [ $start -lt $duration ]; do
    output_file=$(printf "${output_dir}/${base}_5fps_part%03d.mp4" $count)
    ffmpeg -y -ss $start -t $segment_length -i "$downsampled" -c:v libx264 -c:a aac "$output_file"
    start=$((start + step))
    count=$((count + 1))
  done

  echo "Done with $mp4file."
done
