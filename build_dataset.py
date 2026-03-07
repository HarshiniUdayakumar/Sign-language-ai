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
def get_next_index(folder, prefix):

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

        frame = cv2.resize(frame, (640,480))

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

        # POSE
        pose_indices = [0,11,12,13,14]

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

    # FIX FRAME SIZE
    if num_frames >= TARGET_FRAMES:

        indices = np.linspace(0, num_frames-1, TARGET_FRAMES).astype(int)

        sequence = sequence[indices]

    else:

        padding = np.zeros((TARGET_FRAMES-num_frames, sequence.shape[1]))

        sequence = np.vstack((sequence, padding))

    return sequence


# -------- PATHS --------
VIDEO_ROOT = "ASLLVD"
DATA_ROOT = "data"


# -------- HELLO (videos in root) --------
hello_output = os.path.join(DATA_ROOT, "hello")
os.makedirs(hello_output, exist_ok=True)

hello_index = get_next_index(hello_output, "hello")

for file in os.listdir(VIDEO_ROOT):

    if file.startswith("hello") and file.endswith(".mov"):

        video_path = os.path.join(VIDEO_ROOT, file)

        print("Processing HELLO:", file)

        seq = extract_sequence(video_path)

        save_path = os.path.join(hello_output, f"hello_{hello_index}.npy")

        np.save(save_path, seq)

        print("Saved:", save_path)

        hello_index += 1


# -------- OTHER ACTIONS --------
actions = ["thankyou", "yes", "beautiful"]

for action in actions:

    video_folder = os.path.join(VIDEO_ROOT, action)

    output_folder = os.path.join(DATA_ROOT, action)

    os.makedirs(output_folder, exist_ok=True)

    index = get_next_index(output_folder, action)

    for file in os.listdir(video_folder):

        if file.endswith(".mov"):

            video_path = os.path.join(video_folder, file)

            print(f"Processing {action.upper()}:", file)

            seq = extract_sequence(video_path)

            save_path = os.path.join(output_folder, f"{action}_{index}.npy")

            np.save(save_path, seq)

            print("Saved:", save_path)

            index += 1


print("Dataset build complete.")