import cv2
import mediapipe as mp
import numpy as np
import time
import socket
from collections import deque


def connect_to_game():
    while True:
        try:
            c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            c.connect(("127.0.0.1", 5555))
            print("[CV] Connected to game!")
            return c
        except ConnectionRefusedError:
            print("[CV] Game not ready, retrying in 1s...")
            time.sleep(1)

client = connect_to_game()


def send_command(cmd):
    global client
    try:
        client.sendall(cmd.encode())
        print(f"[CV] Sent: {cmd}")
    except Exception as e:
        print(f"[CV] Send failed: {e} — reconnecting...")
        client = connect_to_game()

import os

base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, "models", "pose_landmarker_lite.task")

BaseOptions           = mp.tasks.BaseOptions
PoseLandmarker        = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),
    (11,12),(11,13),(13,15),(15,17),(15,19),(15,21),
    (12,14),(14,16),(16,18),(16,20),(16,22),
    (11,23),(12,24),(23,24),(23,25),(24,26),
    (25,27),(27,29),(29,31),(26,28),(28,30),(30,32)
]

PUNCH_EXTEND_THRESH  = 155
PUNCH_RETRACT_THRESH = 110
PUNCH_SPEED_THRESH   = 200
BLOCK_WRIST_NOSE_Y   = 0.0
BLOCK_HOLD_FRAMES    = 6
COOLDOWN_S           = 0.4

last_action_time = 0.0
ACTION_COOLDOWN  = 0.3


def calculate_angle(p1, p2, p3):
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    cos_t = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0)))


def lm_xy(lm, w, h):
    return int(lm.x * w), int(lm.y * h)


def lm_visible(lm, thresh=0.5):
    return lm.visibility > thresh


def draw_landmarks(image, landmarks):
    h, w = image.shape[:2]
    for a, b in POSE_CONNECTIONS:
        if a < len(landmarks) and b < len(landmarks):
            cv2.line(image, lm_xy(landmarks[a], w, h),
                     lm_xy(landmarks[b], w, h), (0, 255, 0), 2)
    for lm in landmarks:
        cv2.circle(image, lm_xy(lm, w, h), 4, (0, 0, 255), -1)


class ArmTracker:
    HISTORY = 6

    def __init__(self, label, shoulder_idx, elbow_idx, wrist_idx):
        self.label        = label
        self.s_idx        = shoulder_idx
        self.e_idx        = elbow_idx
        self.w_idx        = wrist_idx
        self.angle_hist   = deque(maxlen=self.HISTORY)
        self.time_hist    = deque(maxlen=self.HISTORY)
        self.state        = "idle"
        self.last_punch_t = 0.0
        self.last_speed   = 0.0
        self.peak_speed   = 0.0
        self.block_frames = 0
        self.blocking     = False
        self.just_blocked = False

    def update(self, landmarks, now):
        s = landmarks[self.s_idx]
        e = landmarks[self.e_idx]
        w = landmarks[self.w_idx]

        if not all(lm_visible(x) for x in [s, e, w]):
            return None, None

        angle = calculate_angle((s.x, s.y), (e.x, e.y), (w.x, w.y))
        self.angle_hist.append(angle)
        self.time_hist.append(now)

        speed = 0.0
        if len(self.angle_hist) >= 3:
            da = self.angle_hist[-1] - self.angle_hist[0]
            dt = self.time_hist[-1]  - self.time_hist[0]
            speed = abs(da / dt) if dt > 0 else 0.0
        self.last_speed = speed

        punch_detected = False

        if self.state == "idle" and angle < PUNCH_RETRACT_THRESH:
            self.state      = "extending"
            self.peak_speed = 0.0

        if self.state == "extending":
            self.peak_speed = max(self.peak_speed, speed)
            if angle >= PUNCH_EXTEND_THRESH:
                self.state = "extended"

        if self.state == "extended" and angle < PUNCH_RETRACT_THRESH:
            if (self.peak_speed >= PUNCH_SPEED_THRESH and
                    now - self.last_punch_t > COOLDOWN_S):
                punch_detected    = True
                self.last_punch_t = now
            self.state = "idle"

        return angle, punch_detected

    def update_block(self, wrist_lm, nose_lm):
        if not (lm_visible(wrist_lm) and lm_visible(nose_lm)):
            self.block_frames = 0
            self.just_blocked = False
            self.blocking     = False
            return

        if wrist_lm.y < nose_lm.y + BLOCK_WRIST_NOSE_Y:
            self.block_frames += 1
        else:
            self.block_frames = max(0, self.block_frames - 1)

        was_blocking      = self.blocking
        self.blocking     = self.block_frames >= BLOCK_HOLD_FRAMES
        self.just_blocked = self.blocking and not was_blocking


def speed_color(speed):
    t = min(1.0, speed / 800.0)
    return (0, int(255 * (1 - t)), int(255 * t))


def draw_hud(frame, left, right, l_angle, r_angle):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (20, 10, 35), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    def badge(label, angle, x, blocked, speed):
        col = speed_color(speed)
        cv2.putText(frame, f"{label}: {angle:.0f}deg", (x, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        status = "BLOCKING" if blocked else f"spd:{speed:.0f}"
        scol   = (0, 220, 255) if blocked else (180, 180, 180)
        cv2.putText(frame, status, (x, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, scol, 1)

    if l_angle is not None:
        badge("L-elbow", l_angle, 10,       left.blocking,  left.last_speed)
    if r_angle is not None:
        badge("R-elbow", r_angle, w // 2 + 10, right.blocking, right.last_speed)

    max_speed = max(left.last_speed, right.last_speed)
    bar_w = int(w * min(1.0, max_speed / 900.0))
    if bar_w > 0:
        cv2.rectangle(frame, (0, h - 6), (bar_w, h), speed_color(max_speed), -1)

    cv2.putText(frame, "PUNCH: extend arm fast  |  BLOCK: raise wrist above nose  |  Q: quit",
                (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)


options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE
)

cap = cv2.VideoCapture(1)   # change to 0 if wrong camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

left_arm  = ArmTracker("L", 11, 13, 15)
right_arm = ArmTracker("R", 12, 14, 16)

print("[CV] Running. Punch or block in front of the camera!")
print("[CV] Press Q in this window to quit.")

with PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        now    = time.time()
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = landmarker.detect(mp.Image(mp.ImageFormat.SRGB, rgb))

        l_angle = r_angle = None

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            draw_landmarks(frame, landmarks)
            nose = landmarks[0]

            l_angle, l_punch = left_arm.update(landmarks, now)
            left_arm.update_block(landmarks[15], nose)
            if l_punch:
                send_command("PUNCH")

            r_angle, r_punch = right_arm.update(landmarks, now)
            right_arm.update_block(landmarks[16], nose)
            if r_punch:
                send_command("PUNCH")

            if left_arm.just_blocked or right_arm.just_blocked:
                send_command("BLOCK")

        draw_hud(frame, left_arm, right_arm, l_angle, r_angle)
        cv2.imshow("Pose Controller", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()