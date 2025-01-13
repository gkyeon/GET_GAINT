import cv2
import mediapipe as mp


# MediaPipe Pose 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# 이미지 로드
image = cv2.imread("frame5_0054.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 관절 추출
results = pose.process(image_rgb)

# 관절 시각화
if results.pose_landmarks:
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

# 결과 저장
cv2.imwrite("output_poseframe5_0054.jpg", image)
print("관절 추출 완료")