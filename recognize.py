import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

gesture_model = load_model("models/gesture_model.h5")
emotion_model = load_model("models/emotion_model.h5")
gesture_labels = np.load("models/label_encoder.npy", allow_pickle=True)
emotion_labels = np.array(["happy", "sad", "angry", "neutral", "surprise", "fear", "disgust"])

def preprocess_face_image(frame, face_landmarks, img_size=(48, 48)):
    h, w = frame.shape[:2]
    x_min, y_min, x_max, y_max = w, h, 0, 0
    for lm in face_landmarks.landmark:
        x, y = int(lm.x * w), int(lm.y * h)
        x_min = min(x_min, x)
        x_max = max(x_max, x)
        y_min = min(y_min, y)
        y_max = max(y_max, y)
    x_min = max(0, x_min - 20)
    y_min = max(0, y_min - 20)
    x_max = min(w, x_max + 20)
    y_max = min(h, y_max + 20)
    face_img = frame[y_min:y_max, x_min:x_max]
    if face_img.size == 0:
        return None
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = cv2.resize(face_img, img_size)
    face_img = face_img.astype("float32") / 255.0
    face_img = face_img.reshape(1, img_size[0], img_size[1], 1)
    return face_img

cap = cv2.VideoCapture(0)
with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5) as hands, \
     mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(frame_rgb)
        face_results = face_mesh.process(frame_rgb)
        gesture_text = "Gesture: None"
        emotion_text = "Emotion: None"
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                landmarks = np.array([landmarks])
                gesture_pred = gesture_model.predict(landmarks)
                gesture_idx = np.argmax(gesture_pred, axis=1)[0]
                gesture_conf = gesture_pred[0][gesture_idx]
                gesture_text = f"Gesture: {gesture_labels[gesture_idx]} ({gesture_conf:.2%})"
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                # mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)
                pass  # Skip drawing landmarks
            face_img = preprocess_face_image(frame, face_landmarks)
            if face_img is not None:
                emotion_pred = emotion_model.predict(face_img)
                emotion_idx = np.argmax(emotion_pred, axis=1)[0]
                emotion_conf = emotion_pred[0][emotion_idx]
                emotion_text = f"Emotion: {emotion_labels[emotion_idx]} ({emotion_conf:.2%})"
        cv2.putText(frame, gesture_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, emotion_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Gesture and Emotion Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()