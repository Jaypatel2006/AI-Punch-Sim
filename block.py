import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Setup the Landmarker Configuration
model_path = 'models/pose_landmarker_heavy.task'

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False # Disabled for better performance
)
detector = vision.PoseLandmarker.create_from_options(options)

# 2. Initialize Camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Starting live detection... Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for mirror effect and get dimensions
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Detect landmarks
    detection_result = detector.detect(mp_image)

    # 3. UPGRADED Live Blocking Logic
    is_blocking = False
    
    if detection_result.pose_landmarks:
        landmarks = detection_result.pose_landmarks[0] 
        
        # Extract relevant landmarks
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_wrist = landmarks[15]
        right_wrist = landmarks[16]
        nose = landmarks[0] # Added nose for face-level reference

        # Draw circles on wrists for visual debugging
        for lm in [left_wrist, right_wrist]:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 8, (255, 0, 0), -1)

        # --- THE ACCURACY CHECKS ---
        
        # 1. Visibility Check: Ensure the camera actually sees the wrists clearly (> 60% confidence)
        if left_wrist.visibility > 0.6 and right_wrist.visibility > 0.6:
            
            # 2. Height Check: Hands must be raised above shoulders
            hands_up = (left_wrist.y < left_shoulder.y) and (right_wrist.y < right_shoulder.y)
            
            # 3. Face Protection: Hands must be somewhat near face level (nose.y)
            # We add 0.15 so you don't have to punch yourself in the nose to trigger it
            near_face = (left_wrist.y < nose.y + 0.15) and (right_wrist.y < nose.y + 0.15)

            # 4. Tight Guard Check (X-axis): Hands must be relatively close together
            # Distance of 0.2 means wrists are within 20% of the screen width of each other
            wrist_distance = abs(left_wrist.x - right_wrist.x)
            guard_tight = wrist_distance < 0.25 

            # 5. Depth Check (Z-axis): Hands must be in front of the body
            # In MediaPipe, smaller Z means closer to the camera
            hands_forward = (left_wrist.z < left_shoulder.z) and (right_wrist.z < right_shoulder.z)

            # If ALL conditions are met, it's a solid block!
            if hands_up and near_face and guard_tight and hands_forward:
                is_blocking = True

    # 4. User Interface Feedback
    if is_blocking:
        cv2.putText(frame, "SOLID BLOCK!", (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
    else:
        cv2.putText(frame, "Guard Down", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Punching Simulator - Live Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()