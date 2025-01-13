import os
from yt_dlp import YoutubeDL

def download_youtube_video(video_url, output_path="video.mp4"):
    ydl_opts = {
        'outtmpl': output_path,
        'format': 'best[ext=mp4]',
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])
    print(f"Downloaded video to {output_path}")

# 실행 예제
video_url = "https://www.youtube.com/shorts/R6HHWrMDGrU"
download_youtube_video(video_url, output_path="downloaded_video5.mp4")

