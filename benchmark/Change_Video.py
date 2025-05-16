import ffmpeg

(
    ffmpeg
    .input('Global_Changes/videos/episode_5.MOV')
    .output('Global_Changes/videos/episode_5_1080p_10fps.mp4', 
           an=None,  # remove audio
           r=10,     # set frame rate to 10
           vf='scale=-1:1080',  # scale to 720p
           vcodec='libx264',  # use H.264 codec
           crf=20,  # Constant Rate Factor (lower = better quality, higher = smaller size)
           preset='medium',  # encoding speed vs compression ratio
           movflags='+faststart'  # enable fast start for web playback
    )
    .run()
)   