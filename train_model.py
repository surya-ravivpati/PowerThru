import pandas as pd
import numpy as np
import tensorflow as tf
import sklearn

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

SEQUENCE_LENGTH = 10

#Load Data

df = pd.read_csv(
    "cramp_dataset.csv",
    header = None,
    names = [
        "timestamp",
        "emg",
        "temp",
        "sweat",
        "accel_x",
        "accel_y",
        "accel_z",
        "accel_mag",
        "label"
    ],
    on_bad_lines = "skip"
)

#confirm labels are ints
df["label"] = df["label"].astype(int)

#sort for time
df = df.sort_values("timestamp").reset_index(drop = True)

#features
df["emg_diff"] = df["emg"].diff().fillna(0)
df["accel_diff"] = df["accel_mag"].diff().fillna(0)

features = [
    "emg",
    "temp",
    "sweat",
    "accel_mag",
    "emg_diff",
    "accel_diff"
]

#time split
split_idx = int(0.8 * len(df))

train_df = df.iloc[:split_idx].copy()
val_df = df.iloc[split_idx:].copy()

#scaling
scaler = StandardScaler()
train_df[features] = scaler.fit_transform(train_df[features])
val_df[features] = scaler.transform(val_df[features])

#sequence builder
def make_sequences(data):
    X, Y = [], []

    for i in range(len(data) - SEQUENCE_LENGTH):
        X.append(data[features].iloc[i:i + SEQUENCE_LENGTH].values)
        Y.append(data["label"].iloc[i + SEQUENCE_LENGTH])

    return np.array(X, dtype = np.float32), np.array(Y, dtype = np.float32)

X_train, Y_train = make_sequences(train_df)
X_val, Y_val = make_sequences(val_df)

print("Train X:", X_train.shape)
print("Val X:", X_val.shape)

#class weights
Y_train_int = Y_train.astype(int)

classes = np.unique(Y_train_int)
weights = compute_class_weight(
    class_weight = "balanced",
    classes = classes,
    y = Y_train_int
)

class_weight = dict(zip(classes, weights))

#model
model = Sequential([
    Bidirectional(
        LSTM(64, return_sequences = True),
        input_shape = (SEQUENCE_LENGTH, len(features))
    ),
    Dropout(0.4),

    Bidirectional(LSTM(32)),
    Dropout(0.3),

    Dense(32, activation = "relu"),
    Dropout(0.2),

    Dense(1, activation = "sigmoid")
])

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3),
    loss = tf.keras.losses.BinaryFocalCrossentropy(),
    metrics = [
        "accuracy",
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall(),
        tf.keras.metrics.AUC()
    ]
)

#early stopping
early_stop = EarlyStopping(
    monitor = "val_auc",
    mode = "max",
    patience = 8,
    restore_best_weights = True
)

#train
history = model.fit(
    X_train, Y_train,
    validation_data = (X_val, Y_val),
    epochs = 50,
    batch_size = 32,
    class_weight = class_weight,
    callbacks = [early_stop],
    shuffle = False
)

#save
model.save("fatigue_bilstm.keras")

