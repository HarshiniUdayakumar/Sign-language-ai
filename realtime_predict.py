import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque, Counter

# -------- LOAD MODEL --------
model = load_model("models/sign_model.keras")

labels = {
    0: "HELLO",
    1: "THANKYOU"
}

TARGET_FRAMES = 45
CONFIDENCE_THRESHOLD = 0.85
SMOOTHING_WINDOW = 5

# -------- NORMALIZATION VALUES --------
# These must match training distribution roughly
# Since we used global normalization, we recompute per sequence
def normalize_sequence(seq):
    seq = np.array(seq)
    mean = np.mean(seq)
    std = np.std(seq) + 1e-8
    return (seq - mean) / std


# -------- MEDIAPIPE SETUP --------
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0)

sequence = []
prediction_history = deque(maxlen=SMOOTHING_WINDOW)
stable_prediction = ""

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = holistic.process(rgb)

    # -------- DRAW LANDMARKS --------
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    frame_landmarks = []

    # -------- LEFT HAND --------
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            frame_landmarks.extend([lm.x, lm.y])
    else:
        frame_landmarks.extend([0] * 42)

    # -------- RIGHT HAND --------
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            frame_landmarks.extend([lm.x, lm.y])
    else:
        frame_landmarks.extend([0] * 42)

    # -------- POSE --------
    pose_indices = [0, 11, 12, 13, 14]

    if results.pose_landmarks:
        for idx in pose_indices:
            lm = results.pose_landmarks.landmark[idx]
            frame_landmarks.extend([lm.x, lm.y])
    else:
        frame_landmarks.extend([0] * 10)

    sequence.append(frame_landmarks)

    if len(sequence) > TARGET_FRAMES:
        sequence = sequence[-TARGET_FRAMES:]

    # -------- PREDICTION --------
    if len(sequence) == TARGET_FRAMES:

        if not results.left_hand_landmarks and not results.right_hand_landmarks:
            stable_prediction = ""
            prediction_history.clear()

        else:

            # NORMALIZE sequence
            normalized_seq = normalize_sequence(sequence)

            input_data = np.expand_dims(normalized_seq, axis=0)

            prediction = model.predict(input_data, verbose=0)

            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction)

            if confidence > CONFIDENCE_THRESHOLD:

                prediction_history.append(predicted_class)

                if len(prediction_history) == SMOOTHING_WINDOW:
                    most_common = Counter(prediction_history).most_common(1)[0][0]
                    stable_prediction = labels[most_common]

            else:
                stable_prediction = ""
                prediction_history.clear()

    # -------- DISPLAY --------
    display_text = stable_prediction if stable_prediction != "" else "NO ACTION"

    cv2.putText(
        frame,
        display_text,
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )

    cv2.imshow("Sign Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()