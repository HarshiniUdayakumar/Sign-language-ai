import cv2
import mediapipe as mp
import numpy as np

mp_holistic = mp.solutions.holistic

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

video_path = "ASLLVD/hello.mov"
cap = cv2.VideoCapture(video_path)

sequence = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = holistic.process(rgb_frame)

    frame_landmarks = []

    # ---- LEFT HAND ----
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            frame_landmarks.extend([lm.x, lm.y])
    else:
        frame_landmarks.extend([0] * 42)  # 21*2

    # ---- RIGHT HAND ----
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            frame_landmarks.extend([lm.x, lm.y])
    else:
        frame_landmarks.extend([0] * 42)

    # ---- SELECTED POSE LANDMARKS ----
    pose_indices = [0, 11, 12, 13, 14]  # nose, shoulders, elbows

    if results.pose_landmarks:
        for idx in pose_indices:
            lm = results.pose_landmarks.landmark[idx]
            frame_landmarks.extend([lm.x, lm.y])
    else:
        frame_landmarks.extend([0] * 10)

    sequence.append(frame_landmarks)

cap.release()
sequence = np.array(sequence)
print("Original shape:", sequence.shape)
# ---- FIXED LENGTH = 30 ----
TARGET_FRAMES = 30

num_frames = sequence.shape[0]

if num_frames >= TARGET_FRAMES:
    indices = np.linspace(0, num_frames - 1, TARGET_FRAMES).astype(int)
    fixed_sequence = sequence[indices]
else:
    # pad with zeros if video shorter than 30 frames
    padding = np.zeros((TARGET_FRAMES - num_frames, sequence.shape[1]))
    fixed_sequence = np.vstack((sequence, padding))

print("Fixed shape:", fixed_sequence.shape)