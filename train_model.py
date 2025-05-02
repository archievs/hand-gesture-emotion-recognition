import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def build_gesture_model(num_classes):
    model = Sequential([
        Dense(128, activation="relu", input_shape=(63,)),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

if __name__ == "__main__":
    data = pd.read_csv("data/gestures.csv")
    X = data.iloc[:, :-1].values
    y = data["gesture"].values
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    np.save("models/label_encoder.npy", label_encoder.classes_)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    model = build_gesture_model(len(label_encoder.classes_))
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
    model.save("models/gesture_model.h5")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {accuracy:.4f}")