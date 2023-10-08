import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)

backends = [
    'opencv',
    'ssd',
    'dlib',
    'mtcnn',
    'retinaface',
    'mediapipe',
    'yolov8',
    'yunet',
]

while True:
    ret, frame = cap.read()
    try:
        if frame is None:
            print("none")
            continue
        result = DeepFace.analyze(frame, actions=['emotion'])
        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(frame,
                    result[0]["dominant_emotion"][:],
                    (50, 50),
                    font, 3,
                    (0, 0, 255),
                    2,
                    cv2.LINE_4)

        cv2.imshow('Original Video', frame)
        cv2.waitKey(5000)
        print(result[0]["emotion"])

        if result[0]["dominant_emotion"][:]:
            cap.release()
            cv2.destroyAllWindows()
            break
    except ValueError:
        pass
