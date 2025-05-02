import cv2
import mediapipe as mp
import numpy as np
import os
import csv
from datetime import datetime

mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

def collect_gesture_data(gesture, num_samples=100):
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5) as hands:
        samples_collected = 0
        data = []
        print(f"Collecting {num_samples} samples for gesture: {gesture}. Press 's' to save, 'q' to quit.")
        while samples_collected < num_samples:
            ret, frame = cap.read()
            if not ret:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                    cv2.putText(frame, f"Sample {samples_collected+1}/{num_samples}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Collect Gesture", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') and results.multi_hand_landmarks:
                data.append(landmarks + [gesture])
                samples_collected += 1
            elif key == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()
    return data

def collect_emotion_data(emotion, num_samples=200):
    cap = cv2.VideoCapture(0)
    with mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
        samples_collected = 0
        emotion_dir = f"data/emotions/{emotion}"
        os.makedirs(emotion_dir, exist_ok=True)
        print(f"Collecting {num_samples} samples for emotion: {emotion}. Press 's' to save, 'q' to quit.")
        while samples_collected < num_samples:
            ret, frame = cap.read()
            if not ret:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)
                    pass  # Skip drawing landmarks
                cv2.putText(frame, f"Sample {samples_collected+1}/{num_samples}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Collect Emotion", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') and results.multi_face_landmarks:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                cv2.imwrite(f"{emotion_dir}/{emotion}_{timestamp}.png", frame)
                samples_collected += 1
            elif key == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    mode = input("Choose mode (1: Gesture, 2: Emotion): ")
    if mode == "1":
        gestures = ["fist", "palm", "one", "two", "three", "four", "hello", "none"]
        all_data = []
        for gesture in gestures:
            gesture_data = collect_gesture_data(gesture)
            all_data.extend(gesture_data)
        if all_data:
            with open("data/gestures.csv", "w", newline="") as f:
                writer = csv.writer(f)
                headers = [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)] + [f"z{i}" for i in range(21)] + ["gesture"]
                writer.writerow(headers)
                writer.writerows(all_data)
    elif mode == "2":
        emotions = ["happy", "sad", "angry", "neutral", "surprise", "fear", "disgust"]
        for emotion in emotions:
            collect_emotion_data(emotion)