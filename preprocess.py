import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data(data_path='data/gestures'):
    X, y = [], []
    gestures = os.listdir(data_path)
    
    for gesture in gestures:
        gesture_path = os.path.join(data_path, gesture)
        for file in os.listdir(gesture_path):
            if file.endswith('.npy'):
                landmarks = np.load(os.path.join(gesture_path, file))
                X.append(landmarks)
                y.append(gesture)
    
    X = np.array(X)
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    return X, y, le

def preprocess_data():
    X, y, le = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, le