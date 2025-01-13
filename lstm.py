import cv2
import mediapipe as mp
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from keras.models import Sequential
from keras.layers import LSTM, Dense

# MediaPipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# 이미지에서 관절 데이터 추출 함수
def extract_keypoints(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    keypoints = []

    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            keypoints.append((landmark.x, landmark.y))
    return keypoints

# LSTM 모델 구축 함수
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(3, activation='softmax'))  # 3가지 동작 클래스를 예측: 정상, 비정상 보행
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# K-평균 군집화 함수
def perform_kmeans(keypoints, n_clusters=4):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(keypoints)
    return kmeans

# 비정상 걷기 탐지 함수
def detect_abnormal_gait(new_keypoints, cluster_centers, threshold=0.5):
    distances = pairwise_distances_argmin_min(new_keypoints, cluster_centers)
    if np.any(distances[1] > threshold):
        return "비정상적인 걷기 패턴 탐지됨"
    else:
        return "정상적인 걷기 패턴"

# 1. 이미지에서 관절 데이터 추출
image = cv2.imread("frame_0166.jpg")  # 실제 이미지 경로로 수정 필요
keypoints = extract_keypoints(image)

# 2. LSTM 모델 학습 (간단한 예시 데이터 사용)
X_train = np.array([keypoints])  # 각 프레임의 keypoints를 시퀀스 형태로 배열화
y_train = np.array([0, 1, 2])  # 예시 레이블: 0=정상 보행, 1=비정상 보행, 2=비정상 보행

# LSTM 모델 학습
lstm_model = build_lstm_model(input_shape=(X_train.shape[1], 2))  # keypoints에 따라 (프레임 수, 관절 수) 입력형태
lstm_model.fit(X_train, y_train, epochs=10)

# 3. K-평균 군집화
keypoints_array = np.array(keypoints)  # keypoints는 (x, y) 좌표 리스트
kmeans = perform_kmeans(keypoints_array)

# 군집의 중심 출력 (각 관절의 대표 위치)
cluster_centers = kmeans.cluster_centers_
print("군집 중심 (왼발, 오른발, 왼쪽 어깨, 오른쪽 어깨 등):", cluster_centers)

# 4. 새로운 데이터에 대해 비정상 걷기 탐지
new_keypoints = np.array([0.5, 0.6])  # 새로운 관절 좌표 예시 (새로운 프레임에서의 좌표)
abnormal_gait_result = detect_abnormal_gait(new_keypoints, cluster_centers)
print(abnormal_gait_result)

# 5. LSTM 모델을 사용한 예측 (새로운 관절 데이터 입력)
new_keypoints_seq = np.array([new_keypoints])  # 새로운 시퀀스 입력 (새로운 프레임)
lstm_prediction = lstm_model.predict(new_keypoints_seq)
print("LSTM 예측 결과:", lstm_prediction)
