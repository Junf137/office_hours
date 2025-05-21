import ffmpeg
import os

GLOBAL_VIDEOS_DIR = "./data/global_changes_videos"

for i in range(6):  # episodes 0 to 5
    input_file = os.path.join(GLOBAL_VIDEOS_DIR, f'episode_{i}.mp4')
    output_file = os.path.join(GLOBAL_VIDEOS_DIR, f'episode_{i}_720p_10fps.mp4')

    print(f'Processing {input_file} -> {output_file}')

    (
        ffmpeg
        .input(input_file)
        .output(
            output_file, 
            an=None,  # remove audio
            r=10,     # set frame rate to 10
            vf='scale=-1:720',  # scale to 720p
            vcodec='libx264',  # use H.264 codec
            crf=28,  # Constant Rate Factor (lower = better quality, higher = smaller size)
            preset='medium',  # encoding speed vs compression ratio
            movflags='+faststart'  # enable fast start for web playback
        )
        .run()
    )

    print("Done processing all episodes.")