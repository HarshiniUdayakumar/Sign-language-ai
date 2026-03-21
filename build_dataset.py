import cv2
import mediapipe as mp
import numpy as np
import os

# -------- MEDIAPIPE --------
mp_holistic = mp.solutions.holistic

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

TARGET_FRAMES = 45


# -------- GET NEXT SAFE INDEX --------
def get_next_index(folder):

    files = [f for f in os.listdir(folder) if f.endswith(".npy")]

    if len(files) == 0:
        return 0

    indices = []

    for f in files:
        try:
            index = int(f.split("_")[1].split(".")[0])
            indices.append(index)
        except:
            pass

    return max(indices) + 1


# -------- EXTRACT LANDMARK SEQUENCE --------
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

        # LEFT HAND
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                frame_landmarks.extend([lm.x, lm.y])
        else:
            frame_landmarks.extend([0]*42)

        # RIGHT HAND
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                frame_landmarks.extend([lm.x, lm.y])
        else:
            frame_landmarks.extend([0]*42)

        # POSE (selected points)
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

    num_frames = sequence.shape[0]

    # -------- FIX FRAME SIZE --------
    if num_frames >= TARGET_FRAMES:
        indices = np.linspace(0, num_frames - 1, TARGET_FRAMES).astype(int)
        sequence = sequence[indices]
    else:
        padding = np.zeros((TARGET_FRAMES - num_frames, sequence.shape[1]))
        sequence = np.vstack((sequence, padding))

    return sequence


# -------- PATHS --------
VIDEO_ROOT = "ASLLVD"
DATA_ROOT = "data"

wrong_video_folder = os.path.join(VIDEO_ROOT, "wrong")
wrong_output_folder = os.path.join(DATA_ROOT, "wrong")

os.makedirs(wrong_output_folder, exist_ok=True)


# -------- PROCESS WRONG --------
wrong_index = get_next_index(wrong_output_folder)

for file in os.listdir(wrong_video_folder):

    if file.endswith((".mp4", ".mov", ".avi")):

        video_path = os.path.join(wrong_video_folder, file)

        print("Processing WRONG:", file)

        seq = extract_sequence(video_path)

        save_path = os.path.join(wrong_output_folder, f"wrong_{wrong_index}.npy")

        np.save(save_path, seq)

        print("Saved:", save_path)

        wrong_index += 1


print("✅ WRONG dataset build complete.")