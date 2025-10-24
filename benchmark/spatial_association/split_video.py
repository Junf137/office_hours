import ffmpeg
import os

def split_video(input_file, timestamps, output_dir=None):
    """
    Splits a video into multiple segments based on the given timestamps.

    Args:
        input_file (str): Path to the input video file.
        timestamps (list of float): List of timestamps (in seconds) to split at.
        output_dir (str, optional): Directory to save the output segments.
    """
    # Ensure timestamps are sorted and unique
    timestamps = sorted(set(timestamps))
    
    # Create output directory if not given
    if output_dir is None:
        output_dir = os.path.dirname(input_file)
    os.makedirs(output_dir, exist_ok=True)

    # Get video duration
    try:
        probe = ffmpeg.probe(input_file)
        duration = float(probe['format']['duration'])
    except Exception as e:
        raise RuntimeError(f"Could not read video metadata: {e}")

    # Add start and end boundaries
    split_points = [0.0] + timestamps + [duration]

    # Create video segments
    for i in range(len(split_points) - 1):
        # start = split_points[i]
        start = 0.0
        end = split_points[i + 1]
        segment_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(input_file))[0]}_part_{i}.mp4")

        print(f"Creating segment {i}: {start:.2f}s - {end:.2f}s")

        (
            ffmpeg
            .input(input_file, ss=start, to=end)
            .output(segment_path, c='copy')  # fast split without re-encoding
            .run(quiet=True, overwrite_output=True)
        )

    print("✅ Splitting complete!")


if __name__ == "__main__":
    # Example usage
    input_video = "./data/global_changes_videos/episode_0_720p_10fps.mp4"
    output_directory = "./benchmark/spatial_association/split_videos"
    split_points = [
        28, 
        50, 
        1*60 + 2,
        1*60 + 20,
        1*60 + 33,
        1*60 + 51,
        2*60 + 3,
        3*60 + 6,
        3*60 + 15,
        3*60 + 41,
        4*60 + 9,
        4*60 + 17,
        4*60 + 37,
        4*60 + 50,
        5*60 + 38,
        5*60 + 50,
        6*60 + 20,
        6*60 + 28,
        6*60 + 53,
        7*60 + 7,
        8*60 + 5,
        8*60 + 22,
        8*60 + 54
    ]
    split_video(input_video, split_points, output_directory)
