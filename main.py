import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import matplotlib.pyplot as plt

model_path = "/Users/dvs/Project/AI/Repo/models/pose_landmarker_heavy.task"

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


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

minz = 10
maxz = -10
min_frame = 0
max_frame = 0
frame_no = 0
min_angle = 180
max_angle = 0

def calculate_angle(p1, p2, p3):
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)

    v1 = p1 - p2   # vector from p2 → p1
    v2 = p3 - p2   # vector from p2 → p3

    
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

   
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    angle = np.degrees(np.arccos(cos_theta))
    return angle


def draw_landmarks(image, landmarks):
    h, w, _ = image.shape
    
    


    for a, b in POSE_CONNECTIONS:
        if a < len(landmarks) and b < len(landmarks):
            pa = landmarks[a]
            pb = landmarks[b]

            ax, ay = int(pa.x * w), int(pa.y * h)
            bx, by = int(pb.x * w), int(pb.y * h)

            cv2.line(image, (ax, ay), (bx, by), (0, 255, 0), 2)

    
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
            landmarks = result.pose_landmarks[0]
            draw_landmarks(frame, landmarks)


            if(landmarks[11].visibility > 0.5 and landmarks[13].visibility>0.5 and landmarks[15].visibility>0.5):
                lm1 = landmarks[11]
                lm2 = landmarks[13]
                lm3 = landmarks[15]
                a = calculate_angle((lm1.x,lm1.y),(lm2.x,lm2.y),(lm3.x,lm3.y))
                if(a<min_angle):
                    min_angle = a
                    min_frame = frame

                if(a>max_angle):
                    max_angle = a
                    max_frame = frame

            if landmarks[13].visibility > 0.5:
                lm = landmarks[13]
                if(lm.z < minz):
                    minz = lm.z
                    
                
                if(lm.z > maxz):
                    maxz = lm.z
                    
                
        
        cv2.imshow("Pose Landmarker ", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame += 1

print(min_angle,max_angle)
print(min_frame,max_frame)

plt.imshow(min_frame)
plt.show()

plt.imshow(max_frame)
plt.show()

cap.release()
cv2.destroyAllWindows()