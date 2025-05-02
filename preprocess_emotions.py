import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split

def preprocess_emotion_data(emotions, img_size=(48, 48)):
    data = []
    labels = []
    for emotion_idx, emotion in enumerate(emotions):
        emotion_dir = f"data/emotions/{emotion}"
        for img_name in os.listdir(emotion_dir):
            img_path = os.path.join(emotion_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, img_size)
                img = img.astype("float32") / 255.0
                data.append(img)
                labels.append(emotion_idx)
    data = np.array(data).reshape(-1, img_size[0], img_size[1], 1)
    labels = np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    emotions = ["happy", "sad", "angry", "neutral", "surprise", "fear", "disgust"]
    X_train, X_test, y_train, y_test = preprocess_emotion_data(emotions)
    np.save("data/emotions/X_train.npy", X_train)
    np.save("data/emotions/X_test.npy", X_test)
    np.save("data/emotions/y_train.npy", y_train)
    np.save("data/emotions/y_test.npy", y_test)