import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import pandas as pd

global img, clean_img, x1, x2, y1, y2, cnt, csv_list

# 변수 초기화 (NameError 방지)
img = None
clean_img = None
x1, y1, x2, y2 = 0, 0, 0, 0
sign_number = 31
cnt = 0

os.makedirs(f'./data/{sign_number}', exist_ok=True)

csv_list = {'image':[], 'xmin':[], 'ymin':[], 'xmax':[], 'ymax':[],'label':[]}

cap = cv2.VideoCapture(0)

def click(event, x, y, flags, param):
    global cnt, x1, x2, y1, y2, clean_img
    # 프레임이 없을 때의 클릭 방어
    if clean_img is None:
        return
        
    if event == cv2.EVENT_LBUTTONDOWN:
        # 가시화 선이 없는 순수 원본(clean_img) 저장
        cv2.imwrite(f'./data/{sign_number}/sign_{sign_number}_{cnt}_h.jpg', clean_img)
        csv_list['image'].append(f'sign_{sign_number}_{cnt}_h.jpg')
        csv_list['xmin'].append(x1)
        csv_list['ymin'].append(y1)
        csv_list['xmax'].append(x2)
        csv_list['ymax'].append(y2)
        csv_list['label'].append(sign_number)
        cnt += 1
        if cnt > 50:
            print('complete')

cv2.namedWindow('Dataset')
cv2.setMouseCallback('Dataset', click)

# MediaPipe Tasks API 모델 로드
try:
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1)
    detector = vision.HandLandmarker.create_from_options(options)
except FileNotFoundError:
    print("Error: 'hand_landmarker.task' 파일이 동일한 경로에 필요합니다.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    
    # 출력용(img)과 저장용(clean_img) 객체 분리
    img = frame.copy()
    clean_img = frame.copy()
    
    img_height, img_width = img.shape[:2]
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result = detector.detect(mp_image)

    if result.hand_landmarks:
        for res in result.hand_landmarks:
            joint = np.zeros((21, 3))
            for j, lm in enumerate(res):
                joint[j] = [lm.x, lm.y, lm.z]

            x1, y1 = tuple((joint.min(axis=0)[:2] * [img.shape[1], img.shape[0]] * 0.95).astype(int))
            x2, y2 = tuple((joint.max(axis=0)[:2] * [img.shape[1], img.shape[0]] * 1.05).astype(int))

            # YOLO 상대 좌표 계산
            x_center = (x1 + x2) / (2.0 * img_width)
            y_center = (y1 + y2) / (2.0 * img_height)
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height

            # 좌표가 잘 구해졌는지 사각형 그려보기 (출력용 화면에만 적용)
            cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=2)
            
            text_yolo = f"Label: {sign_number} | CX:{x_center:.2f} CY:{y_center:.2f} W:{width:.2f} H:{height:.2f}"
            cv2.putText(img, text_yolo, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow('Dataset', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

csv_ = pd.DataFrame(csv_list)
csv_.to_csv(f'data/{sign_number}/capstone_sign_{sign_number}.csv', index=False)
