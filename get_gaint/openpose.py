import cv2
import numpy as np

# 모델 로드
protoFile = "C:/Users/gkyeon/Desktop/산단실험/pose_deploy_linevec.prototxt"
weightsFile = "C:/Users/gkyeon/Desktop/산단실험/pose_iter_5840000.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

# 이미지 로드
image_path = "frame_0166.jpg"
image = cv2.imread(image_path)
height, width, _ = image.shape

# 모델에 입력할 전처리
inpBlob = cv2.dnn.blobFromImage(image, 1.0 / 255, (368, 368), (0, 0, 0), swapRB=False, crop=False)
net.setInput(inpBlob)

# 관절 추출
output = net.forward()

# 관절 위치 계산
nPoints = 15  # 모델에 따라 다름
threshold = 0.1
points = []

for i in range(nPoints):
    probMap = output[0, i, :, :]
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

    x = (width * point[0]) / output.shape[3]
    y = (height * point[1]) / output.shape[2]

    if prob > threshold:
        points.append((int(x), int(y)))
        cv2.circle(image, (int(x), int(y)), 8, (0, 255, 255), -1)
    else:
        points.append(None)

# 이미지 저장
output_path = "output_pose166.jpg"
cv2.imwrite(output_path, image)
print(f"관절이 추출된 이미지를 저장했습니다: {output_path}")
