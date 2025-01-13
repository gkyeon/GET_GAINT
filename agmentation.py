import os
import cv2
import numpy as np
from albumentations import (
    HorizontalFlip, RandomBrightnessContrast, HueSaturationValue, Compose
)
from albumentations.core.composition import OneOf

def augment_image(image):
    """
    이미지에 증강 작업을 수행합니다.
    Args:
        image (numpy array): 입력 이미지.
    Returns:
        numpy array: 증강된 이미지.
    """
    transform = Compose([
        HorizontalFlip(p=0.5),  # 좌우 반전
        OneOf([
            RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
            HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.8),
        ], p=1.0)  # 색상 조정
    ])
    augmented = transform(image=image)
    return augmented['image']

def augment_images(input_dir, output_dir):
    """
    디렉토리에 있는 모든 이미지를 증강하고 저장합니다.
    Args:
        input_dir (str): 원본 이미지가 저장된 디렉토리 경로.
        output_dir (str): 증강된 이미지를 저장할 디렉토리 경로.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file_name)
        if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        # 이미지 읽기
        image = cv2.imread(file_path)
        if image is None:
            print(f"Error: {file_path}는 유효한 이미지 파일이 아닙니다.")
            continue

        # 증강 수행
        augmented_image = augment_image(image)

        # 저장
        output_file_path = os.path.join(output_dir, f"aug_{file_name}")
        cv2.imwrite(output_file_path, augmented_image)
        print(f"Saved augmented image: {output_file_path}")

# 실행 예제
input_dir = "./frames"  # 원본 이미지 디렉토리
output_dir = "./augmented_frames"  # 증강된 이미지 디렉토리
augment_images(input_dir, output_dir)
