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

EYE_AR_THRESHOLD = 0.18 # EAR 임계값
EYE_AR_CONSEC_FRAMES = 30 # 눈이 감긴 연속 프레임 수

COUNTER = 0 # 연속 프레임 카운터
ALARM_ON = False # 알람 상태

def check_drowsiness(ear_value):
    global COUNTER, ALARM_ON

    if ear_value < EYE_AR_THRESHOLD:
        COUNTER += 1

        if COUNTER >= EYE_AR_CONSEC_FRAMES:
            # 졸음 상태로 판단
            ALARM_ON = True
            return True
    else:
        # 눈을 뜨면 카운터 초기화
        COUNTER = 0
        ALARM_ON = False
        return False
    
    return False

LEFT_EYE_ID= list(range(36,42))
RIGHT_EYE_ID=list(range(42,48))

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
           # 랜드마크 좌표를 NumPy 배열로 변환
        shape_np = np.zeros((68, 2), dtype="int")
        for i in range(68):
            shape_np[i] = (shape.part(i).x, shape.part(i).y)

        # 왼쪽 눈과 오른쪽 눈 랜드마크 추출
        left_eye = shape_np[LEFT_EYE_ID]
        right_eye = shape_np[RIGHT_EYE_ID]
        
        # EAR 계산
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0
        cv2.putText(img, "EAR: {:.2f}".format(ear), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        drowsy_state = check_drowsiness(ear)      
        if drowsy_state:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)

        cv2.imshow('face detect', img)
    else:
        break
    if cv2.waitKey(5) == 27:
        break

cv2.destroyAllWindows()