# 기본 랜드마크 검출
import cv2
import dlib
from scipy.spatial import distance as dist
import numpy as np
# 얼굴 검출기와 랜드마크 검출기 생성 --- ①
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
cap = cv2.VideoCapture(0)

# EAR 계산 함수 정의
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

while cap.isOpened():    
    ret, img = cap.read()  # 프레임 읽기
    if ret:

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 얼굴 영역 검출 --- ②
        faces = detector(gray)
        for rect in faces:
            # 얼굴 영역을 좌표로 변환 후 사각형 표시 --- ③
            x,y = rect.left(), rect.top()
            w,h = rect.right()-x, rect.bottom()-y
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)

            # 얼굴 랜드마크 검출 --- ④
            shape = predictor(gray, rect)
            # 왼쪽 눈 랜드마크 (36번 ~ 41번)
            for i in range(36, 42):
                part = shape.part(i)
                
                
            # 오른쪽 눈 랜드마크 (42번 ~ 47번)
            for i in range(42, 48):
                part = shape.part(i)
                
        cv2.imshow('face detect', img)
    else:
        break
    if cv2.waitKey(5) == 27:
        break

cv2.destroyAllWindows()