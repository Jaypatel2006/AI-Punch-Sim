import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = r"D:\college\AI_Project\models\pose_landmarker_heavy.task"

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Pose connections (same as official, but copied manually)
POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),
    (0,4),(4,5),(5,6),(6,8),
    (9,10),
    (11,12),(11,13),(13,15),(15,17),(15,19),(15,21),
    (12,14),(14,16),(16,18),(16,20),(16,22),
    (11,23),(12,24),(23,24),(23,25),(24,26),
    (25,27),(27,29),(29,31),
    (26,28),(28,30),(30,32)
]

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE
)

cap = cv2.VideoCapture(0)

def draw_landmarks(image, landmarks):
    h, w, _ = image.shape

    # Draw connections
    for a, b in POSE_CONNECTIONS:
        if a < len(landmarks) and b < len(landmarks):
            pa = landmarks[a]
            pb = landmarks[b]

            ax, ay = int(pa.x * w), int(pa.y * h)
            bx, by = int(pb.x * w), int(pb.y * h)

            cv2.line(image, (ax, ay), (bx, by), (0, 255, 0), 2)

    # Draw keypoints
    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (x, y), 4, (0, 0, 255), -1)

with PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(mp.ImageFormat.SRGB, rgb)

        result = landmarker.detect(mp_image)

        if result.pose_landmarks:
            draw_landmarks(frame, result.pose_landmarks[0])
        
        cv2.imshow("Pose Landmarker ", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
