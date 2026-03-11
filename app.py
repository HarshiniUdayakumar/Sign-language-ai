import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque, Counter

# -------- SETTINGS --------
TARGET_FRAMES = 45
CONFIDENCE_THRESHOLD = 0.70
SMOOTHING_WINDOW = 4

labels = {
    0: "HELLO",
    1: "THANKYOU",
    2: "YES",
    3: "BEAUTIFUL"
}

# -------- CACHE MODEL --------
@st.cache_resource
def load_my_model():
    return load_model("models/sign_model_4gesture.keras")

model = load_my_model()

# -------- MEDIAPIPE --------
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=0,   # faster
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)

# -------- NORMALIZATION --------
def normalize_sequence(seq):
    seq = np.array(seq)
    seq = (seq - np.mean(seq, axis=0)) / (np.std(seq, axis=0) + 1e-8)
    return seq


# -------- STREAMLIT UI --------
st.set_page_config(
    page_title="Sign Language Recognition",
    page_icon="🤟",
    layout="centered"
)

st.title("🤟 Sign Language Recognition")
st.write("Real-time sign language detection using AI")

st.divider()

run = st.checkbox("Start Camera")

frame_placeholder = st.image([])

prediction_text = st.empty()

sequence = []
prediction_history = deque(maxlen=SMOOTHING_WINDOW)
stable_prediction = ""

frame_counter = 0

camera = cv2.VideoCapture(0)

while run:

    ret, frame = camera.read()

    if not ret:
        st.error("Camera not detected")
        break

    # -------- RESIZE FOR SPEED --------
    frame = cv2.resize(frame, (480, 360))

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

    frame_counter += 1

    # -------- PREDICTION (EVERY 3 FRAMES) --------
    if len(sequence) == TARGET_FRAMES and frame_counter % 3 == 0:

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

    # -------- DISPLAY FRAME --------
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame_placeholder.image(frame)

    if stable_prediction != "":
        prediction_text.success(f"Detected Sign: {stable_prediction}")
    else:
        prediction_text.info("Detected Sign: NO ACTION")

camera.release()