import os
import glob
import subprocess
import re
import shutil


def process_video(video_path: str, target_fps: int):
    """Process a single video file

    Args:
        video_path: Path to the original video
        target_fps: Target frame rate for downsampling
    """
    # Get directory, filename and extension info
    video_dir = os.path.dirname(video_path)
    filename = os.path.basename(video_path)
    file_basename, file_ext = os.path.splitext(filename)
    file_ext = file_ext.lower()

    # Check if this is an episode video using regex pattern
    pattern = re.compile(r"ep_\d+_[ab]\.")
    if not pattern.search(filename.lower()):
        return False  # Skip videos that don't match our pattern

    print(f"Found matching video: {filename}")

    # Create backup folder if it doesn't exist
    backup_dir = os.path.join(video_dir, "backup")
    os.makedirs(backup_dir, exist_ok=True)

    # Move original to backup folder
    backup_path = os.path.join(backup_dir, filename)
    shutil.move(video_path, backup_path)
    print(f"Moved original to {backup_path}")

    # Output will have same name as original, but will be MP4
    output_path = os.path.join(video_dir, f"{file_basename}.mp4")

    # Process the video - convert and downsample in one step
    # Note: -an removes audio, use backup file as input
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",  # Overwrite output file if it exists
        "-loglevel",
        "error",  # Only show errors
        "-i",
        backup_path,  # Use backup file as input
        "-an",  # Remove audio
        "-filter:v",
        f"fps=fps={target_fps}",
        "-c:v",
        "libx264",  # Use H.264 codec
        "-preset",
        "medium",  # Balance between speed and quality
        "-crf",
        "23",  # Quality factor (lower is better)
        output_path,
    ]

    print(f"Processing {filename}: Converting to MP4, removing audio, downsampling to {target_fps} fps...")
    subprocess.run(ffmpeg_cmd, check=True)

    return True


if __name__ == "__main__":
    # Configuration
    ROOT_DIR = "root_folder"  # Change this to your actual root folder path
    TARGET_FPS = 10

    # Find all video files in all subdirectories
    video_extensions = [".mp4", ".mov", ".MP4", ".MOV"]
    all_videos = []

    # Walk through all subdirectories
    for root, _, files in os.walk(ROOT_DIR):
        for file in files:
            if any(file.lower().endswith(ext.lower()) for ext in video_extensions):
                all_videos.append(os.path.join(root, file))

    print(f"Found {len(all_videos)} video files in total.")

    # Process all matching videos
    processed_count = 0
    for video_path in all_videos:
        if process_video(video_path, TARGET_FPS):
            processed_count += 1

    print(f"Done! Processed {processed_count} videos.")
