import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# -------- LOAD DATA --------
DATA_PATH = "data"

X = []
y = []

# Label mapping
labels = {
    "hello": 0,
    "thankyou": 1
}

for label_name, label_index in labels.items():
    folder_path = os.path.join(DATA_PATH, label_name)

    for file in os.listdir(folder_path):
        if file.endswith(".npy"):
            sequence = np.load(os.path.join(folder_path, file))
            X.append(sequence)
            y.append(label_index)

X = np.array(X)
y = to_categorical(y)

print("X shape:", X.shape)
print("y shape:", y.shape)

# -------- TRAIN TEST SPLIT --------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# -------- BUILD LSTM MODEL --------
model = Sequential()

model.add(LSTM(64, return_sequences=True, input_shape=(30, 94)))
model.add(Dropout(0.2))

model.add(LSTM(32))
model.add(Dropout(0.2))

model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -------- TRAIN MODEL --------
history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=2,
    validation_data=(X_test, y_test)
)

# -------- EVALUATE MODEL --------
loss, accuracy = model.evaluate(X_test, y_test)
print("Final Test Accuracy:", accuracy)

# -------- SAVE MODEL --------
model.save("models/sign_model.h5")
print("Model saved successfully!")