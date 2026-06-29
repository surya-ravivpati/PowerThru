import pandas as pd
import numpy as np
import csv
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential


SEQUENCE_LENGTH = 10

df = pd.read_csv("cramp_dataset.csv")

df["accel_mag"] = np.sqrt(
    df["accel_x"]**2 +
    df["accel_y"]**2 +
    df["accel_z"]**2
)

features = [
    "emg",
    "temp",
    "sweat",
    "accel_x",
    "accel_y",
    "accel_z",
    "label"
]

X = []
Y = []

for i in range(len(df) - SEQUENCE_LENGTH):
    sequence = df[features].iloc[
        i:i + SEQUENCE_LENGTH
    ].values

    label = df["label"].iloc[
        i + SEQUENCE_LENGTH
    ]

    X.append(sequence)
    Y.append(label)

X = np.array(X)
Y = np.array(Y)

print(len(df))
print(X.shape)
print(Y.shape)

from tensorflow.keras.layers import (
    Bidirectional,
    LSTM,
    Dense,
    Dropout
)

model = Sequential()

model.add(
    Bidirectional(
        LSTM(
            64,
            return_sequences = True
        ),
        input_shape=(10,7)
    )
)

model.add(
    Dropout(0.3)
)

model.add(
    Bidirectional(
        LSTM(32)
    )
)

model.add(
    Dense(
        16,
        activation = "relu"
    )
)

model.add(
    Dense(
        1,
        activation = "sigmoid"
    )
)

model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

history = model.fit(
    X,
    Y,
    epochs = 50,
    batch_size = 32,
    validation_split = 0.2
)

model.save(
    "fatigue_bilstm.h5"
)