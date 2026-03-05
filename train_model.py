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
    "thankyou": 1
}

# -------- LOAD DATA --------
X = []
y = []

for label_name, label_index in labels.items():
    folder_path = os.path.join(DATA_PATH, label_name)

    for file in os.listdir(folder_path):
        if file.endswith(".npy"):
            sequence = np.load(os.path.join(folder_path, file))

            # Safety check
            if sequence.shape == (TARGET_FRAMES, FEATURES):
                X.append(sequence)
                y.append(label_index)
            else:
                print(f"Skipped {file} due to wrong shape {sequence.shape}")

X = np.array(X)
y = to_categorical(y)

print("X shape:", X.shape)
print("y shape:", y.shape)

if len(X) == 0:
    raise ValueError("No valid data found. Check dataset.")

# -------- TRAIN TEST SPLIT (Stratified) --------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# -------- BUILD MODEL --------
model = Sequential()

model.add(Input(shape=(TARGET_FRAMES, FEATURES)))

# First LSTM
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.4))

# Second LSTM
model.add(LSTM(32))
model.add(Dropout(0.4))

# Dense layers
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -------- EARLY STOPPING --------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=7,
    restore_best_weights=True
)

# -------- TRAIN --------
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=4,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# -------- EVALUATE --------
loss, accuracy = model.evaluate(X_test, y_test)
print("Final Test Accuracy:", accuracy)

# -------- SAVE MODEL --------
os.makedirs("models", exist_ok=True)
model.save("models/sign_model.keras")

print("Model saved successfully!")