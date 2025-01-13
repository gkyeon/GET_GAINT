import cv2
import os

def extract_frames(video_path, output_dir, frame_rate=3):
    """
    비디오에서 초당 지정된 수의 프레임을 추출하여 저장합니다.

    Args:
        video_path (str): 입력 비디오 파일 경로.
        output_dir (str): 저장할 이미지 디렉토리 경로.
        frame_rate (int): 초당 추출할 프레임 수.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 비디오 파일 열기
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: 비디오 파일을 열 수 없습니다.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)  # 비디오의 초당 프레임 수
    frame_interval = int(fps / frame_rate)  # 프레임 간격 계산

    frame_count = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 지정된 간격마다 프레임 저장
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"총 {saved_count}개의 프레임이 {output_dir}에 저장되었습니다.")

# 실행 예제
video_path = "downloaded_video5.mp4"  # 비디오 파일 경로
output_dir = "./frames5"  # 저장할 디렉토리
extract_frames(video_path, output_dir, frame_rate=3)  # 초당 3프레임 추출
