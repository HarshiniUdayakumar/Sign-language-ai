import numpy as np
import os

DATA_PATH = "data"
TARGET_FRAMES = 45

def resample_sequence(sequence, target_frames):
    original_frames = sequence.shape[0]
    indices = np.linspace(0, original_frames - 1, target_frames).astype(int)
    return sequence[indices]

for label in ["hello", "thankyou"]:
    folder = os.path.join(DATA_PATH, label)

    for file in os.listdir(folder):
        if file.endswith(".npy"):
            path = os.path.join(folder, file)
            seq = np.load(path)

            if seq.shape[0] == 30:
                print(f"Fixing {file}")
                new_seq = resample_sequence(seq, TARGET_FRAMES)
                np.save(path, new_seq)

print("All 30-frame files converted to 45 frames.")