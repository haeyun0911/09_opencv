import cv2
import numpy as np
import os, glob

# 변수 설정 ---①
base_dir = './faces'
min_accuracy = 85

# LBP 얼굴 인식기 및 케스케이드 얼굴 검출기 생성 및 훈련 모델 읽기 ---②
face_classifier = cv2.CascadeClassifier(
                 '../data/haarcascade_frontalface_default.xml')
model = cv2.face.LBPHFaceRecognizer_create()
model.read(os.path.join(base_dir, 'all_face.xml'))

# 디렉토리 이름으로 사용자 이름과 아이디 매핑 정보 생성 ---③
dirs = [d for d in glob.glob(base_dir+"/*") if os.path.isdir(d)]
names = dict([])
for dir in dirs:
    dir = os.path.basename(dir)
    name, id = dir.split('_')
    names[int(id)] = name

# 카메라 캡처 장치 준비 
cap = cv2.VideoCapture(0)
# 이전 프레임의 얼굴 정보를 저장할 변수
prev_faces = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("no frame")
        break
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    # 현재 프레임에서 얼굴 검출 ---④
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    # 감지된 얼굴이 있을 경우
    if len(faces) > 0:
        prev_faces = faces
        for (x,y,w,h) in faces:
            # 얼굴 영역 표시 및 인식 결과 출력 (기존 로직)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            label, confidence = model.predict(face)
            
            if confidence < 400:
                accuracy = int( 100 * (1 -confidence/400))
                if accuracy >= min_accuracy:
                    msg = '%s(%.0f%%)'%(names[label], accuracy)
                else:
                    msg = 'Unknown'
            else:
                msg = 'Unknown'
            
            txt, base = cv2.getTextSize(msg, cv2.FONT_HERSHEY_PLAIN, 1, 3)
            cv2.rectangle(frame, (x,y-base-txt[1]), (x+txt[0], y+txt[1]), \
                          (0,255,255), -1)
            cv2.putText(frame, msg, (x, y), cv2.FONT_HERSHEY_PLAIN, 1, \
                        (200,200,200), 2,cv2.LINE_AA)
    
    # 현재 프레임에 얼굴이 감지되지 않았지만 이전 프레임에 얼굴이 있었을 경우
    elif len(prev_faces) > 0:
        for (x,y,w,h) in prev_faces:
            # 이전 위치에 빨간색 사각형과 메시지 출력
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2) # 빨간색 사각형
            msg = 'Face Lost'
            txt, base = cv2.getTextSize(msg, cv2.FONT_HERSHEY_PLAIN, 1, 3)
            cv2.rectangle(frame, (x,y-base-txt[1]), (x+txt[0], y+txt[1]), \
                          (0,0,255), -1) # 빨간색 배경
            cv2.putText(frame, msg, (x, y), cv2.FONT_HERSHEY_PLAIN, 1, \
                        (200,200,200), 2,cv2.LINE_AA)
        
        # 이전 얼굴 정보 초기화 (한 번만 표시되도록)
        # prev_faces = []
        
    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) == 27: #esc 
        break

cap.release()
cv2.destroyAllWindows()