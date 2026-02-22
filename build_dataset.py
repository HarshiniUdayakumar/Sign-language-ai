import cv2
import mediapipe as mp
import numpy as np
import os

# -------- MEDIAPIPE SETUP --------
mp_holistic = mp.solutions.holistic

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

TARGET_FRAMES = 30


# -------- FUNCTION TO EXTRACT SEQUENCE --------
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

        # ---- LEFT HAND ----
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                frame_landmarks.extend([lm.x, lm.y])
        else:
            frame_landmarks.extend([0] * 42)

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

    # ---- FIX TO 30 FRAMES ----
    num_frames = sequence.shape[0]

    if num_frames >= TARGET_FRAMES:
        indices = np.linspace(0, num_frames - 1, TARGET_FRAMES).astype(int)
        sequence = sequence[indices]
    else:
        padding = np.zeros((TARGET_FRAMES - num_frames, sequence.shape[1]))
        sequence = np.vstack((sequence, padding))

    return sequence


# -------- PROCESS HELLO --------
HELLO_FOLDER = "ASLLVD"
HELLO_OUTPUT = "data/hello"
os.makedirs(HELLO_OUTPUT, exist_ok=True)

hello_index = 0

for file in os.listdir(HELLO_FOLDER):
    if file.startswith("hello") and file.endswith(".mov"):
        video_path = os.path.join(HELLO_FOLDER, file)
        print(f"Processing HELLO {file}...")

        seq = extract_sequence(video_path)

        save_path = os.path.join(HELLO_OUTPUT, f"hello_{hello_index}.npy")
        np.save(save_path, seq)

        print("Saved:", save_path)
        hello_index += 1


# -------- PROCESS THANKYOU --------
THANK_FOLDER = "ASLLVD/thankyou"
THANK_OUTPUT = "data/thankyou"
os.makedirs(THANK_OUTPUT, exist_ok=True)

thank_index = 0

for file in os.listdir(THANK_FOLDER):
    if file.endswith(".mov"):
        video_path = os.path.join(THANK_FOLDER, file)
        print(f"Processing THANKYOU {file}...")

        seq = extract_sequence(video_path)

        save_path = os.path.join(THANK_OUTPUT, f"thank_{thank_index}.npy")
        np.save(save_path, seq)

        print("Saved:", save_path)
        thank_index += 1

print("Dataset build complete.")