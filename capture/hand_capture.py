import cv2
import mediapipe as mp

# Initialize MediaPipe
mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Setup face and hands
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Webcam
cap = cv2.VideoCapture(0)

# Important face landmark indices (example)
important_face_indices = [33, 133, 362, 263, 78, 308, 13, 14, 87, 317]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, _ = frame.shape

    # Face landmarks
    face_results = face_mesh.process(rgb_frame)
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            # Draw eyes
            eyes_indices = [33, 133, 160, 159, 158, 144, 153, 154, 155,
                            263, 362, 387, 386, 385, 373, 380, 374, 381]
            for idx in eyes_indices:
                lm = face_landmarks.landmark[idx]
                x, y = int(lm.x * width), int(lm.y * height)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            # Draw lips
            lips_indices = list(range(61, 88)) + list(range(291, 317))
            for idx in lips_indices:
                lm = face_landmarks.landmark[idx]
                x, y = int(lm.x * width), int(lm.y * height)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            # Optional: connect key points for contour
            cv2.line(frame,
                     (int(face_landmarks.landmark[61].x*width), int(face_landmarks.landmark[61].y*height)),
                     (int(face_landmarks.landmark[78].x*width), int(face_landmarks.landmark[78].y*height)),
                     (0, 255, 0), 1)

    # Hand landmarks
    hand_results = hands.process(rgb_frame)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            # Draw skeleton
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
            )
            # Highlight fingertips
            for tip_idx in [4, 8, 12, 16, 20]:
                lm = hand_landmarks.landmark[tip_idx]
                x, y = int(lm.x * width), int(lm.y * height)
                cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)  # red fingertips

    cv2.imshow("Face + Hand Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()