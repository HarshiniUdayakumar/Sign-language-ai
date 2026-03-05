import cv2
import mediapipe as mp
import numpy as np
import os

# -------- SETTINGS --------
DATA_PATH = "data"
TARGET_FRAMES = 45

ACTIONS = {
    ord('h'): "hello",
    ord('t'): "thankyou",
    ord('y'): "yes"
}

# -------- MEDIAPIPE --------
mp_holistic = mp.solutions.holistic

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -------- CREATE FOLDERS --------
for action in ACTIONS.values():
    os.makedirs(os.path.join(DATA_PATH, action), exist_ok=True)

# -------- COUNT EXISTING FILES --------
def get_next_index(action):

    folder = os.path.join(DATA_PATH, action)

    existing = [f for f in os.listdir(folder) if f.endswith(".npy")]

    return len(existing)

# -------- EXTRACT LANDMARKS --------
def extract_landmarks(results):

    frame_landmarks = []

    # LEFT HAND
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            frame_landmarks.extend([lm.x, lm.y])
    else:
        frame_landmarks.extend([0] * 42)

    # RIGHT HAND
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            frame_landmarks.extend([lm.x, lm.y])
    else:
        frame_landmarks.extend([0] * 42)

    # POSE
    pose_indices = [0, 11, 12, 13, 14]

    if results.pose_landmarks:
        for idx in pose_indices:
            lm = results.pose_landmarks.landmark[idx]
            frame_landmarks.extend([lm.x, lm.y])
    else:
        frame_landmarks.extend([0] * 10)

    return frame_landmarks


# -------- START CAMERA --------
cap = cv2.VideoCapture(0)

print("Press 'h' for HELLO")
print("Press 't' for THANKYOU")
print("Press 'y' for YES")
print("Press 'q' to quit")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = holistic.process(rgb)

    cv2.putText(frame,
                "Press h / t / y to record",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2)

    cv2.imshow("Recorder", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

    if key in ACTIONS:

        action = ACTIONS[key]

        print(f"\nRecording {action}...")

        sequence = []
        frame_count = 0

        while frame_count < TARGET_FRAMES:

            ret, frame = cap.read()

            frame = cv2.resize(frame, (640, 480))

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = holistic.process(rgb)

            landmarks = extract_landmarks(results)

            sequence.append(landmarks)

            cv2.putText(frame,
                        f"Recording {action} {frame_count+1}/{TARGET_FRAMES}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2)

            cv2.imshow("Recorder", frame)

            cv2.waitKey(1)

            frame_count += 1

        sequence = np.array(sequence)

        index = get_next_index(action)

        save_path = os.path.join(DATA_PATH, action, f"{action}_{index}.npy")

        np.save(save_path, sequence)

        print(f"Saved: {save_path}")

cap.release()

cv2.destroyAllWindows()