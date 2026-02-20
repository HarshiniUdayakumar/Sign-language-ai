import cv2
import mediapipe as mp
import numpy as np
import os

mp_holistic = mp.solutions.holistic

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

VIDEO_FOLDER = "ASLLVD"
OUTPUT_FOLDER = "data/hello"
TARGET_FRAMES = 30

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def extract_sequence(video_path):
    cap = cv2.VideoCapture(video_path)
    sequence = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        frame_landmarks = []

        # Left Hand
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                frame_landmarks.extend([lm.x, lm.y])
        else:
            frame_landmarks.extend([0]*42)

        # Right Hand
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                frame_landmarks.extend([lm.x, lm.y])
        else:
            frame_landmarks.extend([0]*42)

        # Selected Pose Landmarks
        pose_indices = [0, 11, 12, 13, 14]
        if results.pose_landmarks:
            for idx in pose_indices:
                lm = results.pose_landmarks.landmark[idx]
                frame_landmarks.extend([lm.x, lm.y])
        else:
            frame_landmarks.extend([0]*10)

        sequence.append(frame_landmarks)

    cap.release()

    sequence = np.array(sequence)

    # ---- FIX LENGTH ----
    num_frames = sequence.shape[0]

    if num_frames >= TARGET_FRAMES:
        indices = np.linspace(0, num_frames-1, TARGET_FRAMES).astype(int)
        sequence = sequence[indices]
    else:
        padding = np.zeros((TARGET_FRAMES - num_frames, sequence.shape[1]))
        sequence = np.vstack((sequence, padding))

    return sequence


# Process all videos
for i, file in enumerate(os.listdir(VIDEO_FOLDER)):
    if file.endswith(".mov"):
        video_path = os.path.join(VIDEO_FOLDER, file)
        print(f"Processing {file}...")
        seq = extract_sequence(video_path)

        save_path = os.path.join(OUTPUT_FOLDER, f"hello_{i}.npy")
        np.save(save_path, seq)

        print("Saved:", save_path)

print("Dataset build complete.")