import cv2
import mediapipe as mp
import numpy as np
import joblib

# Modeli yükle
svm_model = joblib.load("svm_winner.pkl")

# Mediapipe el algılama tanımlayıcıları
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Kamera başlat
cap = cv2.VideoCapture(0)

# Video kaydı başlat
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output.mp4", fourcc, 30, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark])
            wrist_x, wrist_y, wrist_z = landmarks[0]
            landmarks[:, 0] -= wrist_x
            landmarks[:, 1] -= wrist_y
            mid_finger_x, mid_finger_y, _ = landmarks[12]
            scale_factor = np.sqrt(mid_finger_x**2 + mid_finger_y**2)
            landmarks[:, 0] /= scale_factor
            landmarks[:, 1] /= scale_factor

            features = landmarks.flatten().reshape(1, -1)
            prediction = svm_model.predict(features)[0]

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.putText(frame, f'Prediction: {prediction}', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    out.write(frame)
    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
