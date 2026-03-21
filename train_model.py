import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# -------- SETTINGS --------
DATA_PATH = "data"
TARGET_FRAMES = 45
FEATURES = 94

labels = {
    "hello": 0,
    "thankyou": 1,
    "yes": 2,
    "beautiful": 3,
    "wrong": 4
}

# -------- LOAD DATA --------
X = []
y_raw = []

for label_name, label_index in labels.items():

    folder_path = os.path.join(DATA_PATH, label_name)

    if not os.path.exists(folder_path):
        print(f"Missing {folder_path}")
        continue

    files = [f for f in os.listdir(folder_path) if f.endswith(".npy")]

    print(f"{label_name}: {len(files)} samples")

    # ⚠ limit wrong to reduce dominance
    if label_name == "wrong":
        files = files[:40]

    for file in files:

        sequence = np.load(os.path.join(folder_path, file))

        if sequence.shape == (TARGET_FRAMES, FEATURES):
            X.append(sequence)
            y_raw.append(label_index)

X = np.array(X)
y_raw = np.array(y_raw)

y = to_categorical(y_raw)

print("X:", X.shape)
print("y:", y.shape)

# -------- SPLIT --------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y_raw
)

# -------- MODEL --------
model = Sequential()
model.add(Input(shape=(TARGET_FRAMES, FEATURES)))

model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(32))
model.add(Dropout(0.3))

model.add(Dense(32, activation='relu'))
model.add(Dense(len(labels), activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -------- TRAIN --------
early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=4,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# -------- SAVE --------
os.makedirs("models", exist_ok=True)
model.save("models/sign_model_fixed.keras")
# -------- EVALUATE --------
loss, accuracy = model.evaluate(X_test, y_test)

print("\n🎯 Final Test Accuracy:", accuracy)

print("Model trained successfully!")