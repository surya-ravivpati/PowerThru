import json
import time
import csv
import os
import random
DATASET_FILE = "cramp_dataset.csv"
label = -1
MODE = False

HEADER = [
    "timestamp",
    "emg",
    "temp",
    "sweat",
    "accel_x",
    "accel_y",
    "accel_z",
    "accel_mag",
    "score",
    "label"
]

def calculate_score(emg, temp, sweat, ax, ay, az, am):
    emg_norm = emg
    temp_norm = (temp - 36) / 3
    temp_norm = max(0, min(1, temp_norm))
    sweat_norm = sweat 
    accel_norm = am / 3
    accel_norm = max(0, min(1, accel_norm))

    risk = (
        0.4 * emg_norm +
        0.25 * sweat_norm +
        0.2 * temp_norm +
        0.15 * accel_norm
    )

    score = int(risk * 100)

    return score

def score_to_label(score):
    
    if score < 60:
        return 0
    return 1

if not os.path.exists(DATASET_FILE):
    with open(DATASET_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)

while MODE == True:
    emg = float(input("EMG: "))
    temp = float(input("Temp: "))
    sweat = float(input("Sweat: "))
    ax = float(input("Accel X: "))
    ay = float(input("Accel Y: "))
    az = float(input("Accel Z: "))

    print("\nLabel:")
    print("0 = Normal")
    print("1 = Cramp/Fatigue Event")

    label = int(input("Label: "))

    am = (ax**2 + ay**2 + az**2)**0.5

    with open(DATASET_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            time.time(),
            emg,
            temp,
            sweat,
            ax,
            ay,
            az,
            am,
            label
        ])
    print("Saved.\n")
else:
    emg = random.uniform(0,1)
    temp = random.uniform(36,39)
    sweat = random.uniform(0,1)
    ax = random.uniform(-2,2)
    ay = random.uniform(-2,2)
    az = random.uniform(-2,2)

    am = (ax**2 + ay**2 + az**2)**0.5
    
    score = calculate_score(emg,temp,sweat,ax,ay,az,am)
    label = score_to_label(score)
    with open(DATASET_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            time.time(),
            round(emg,2),
            round(temp,2),
            round(sweat,2),
            round(ax,2),
            round(ay,2),
            round(az,2),
            round(am,2),
            score,
            label
        ])
    print("Saved.\n")
    print(f"Score = ", score)
    print(f"Label =", label)

    time.sleep(0.05)

