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
        result = DeepFace.analyze(frame, actions=['emotion'], detector_backend=backends[4])
    except ValueError:
        pass

    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(frame,
                result[0]["dominant_emotion"][:],
                (50, 50),
                font, 3,
                (0, 0, 255),
                2,
                cv2.LINE_4)
    cv2.imshow('Original Video', frame)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()