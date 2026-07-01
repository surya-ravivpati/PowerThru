import csv
import time
import os
import random
import math

DATASET_FILE = "cramp_dataset.csv"

HEADER = [
    "timestamp",
    "emg",
    "temp",
    "sweat",
    "accel_x",
    "accel_y",
    "accel_z",
    "accel_mag",
    "label",
]

if not os.path.exists(DATASET_FILE):
    with open(DATASET_FILE, "w", newline = "") as f:
        csv.writer(f).writerow(HEADER)

#sim state
emg = 0.3
temp = 37.0
sweat = 0.2
fatigue_state = 0 # 0 = norm, 1 = cramp possibilty

def clamp(x, a, b):
    return max(a, min(b, x))

def update_state():
    global emg, temp, sweat, fatigue_state

    #occasional cramp
    if random.random() < 0.01:
        fatigue_state = 1
    elif random.random() < 0.05:
        fatigue_state = 0
    
    #usual drift
    emg += random.uniform(-0.02, 0.03)
    temp += random.uniform(-0.01, 0.02)
    sweat += random.uniform(-0.02, 0.03)

    #cramp spike behavior
    if fatigue_state == 1:
        emg += random.uniform(0.05, 0.15)
        sweat += random.uniform(0.05, 0.12)
        temp += random.uniform(0.02, 0.08)
    
    emg = clamp(emg, 0, 1)
    temp = clamp(temp, 35, 40)
    sweat = clamp(sweat, 0, 1)

def generate_accel():
    ax = random.uniform(-1,1)
    ay = random.uniform(-1,1)
    az = random.uniform(-1,1)

    if fatigue_state == 1:
        ax += random.uniform(-1,1)
        ay += random.uniform(-1,1)
        az += random.uniform(-1,1)

    am = math.sqrt(ax**2 + ay**2 + az**2)
    return ax,ay,az,am

def label_from_state():
    return fatigue_state

#data loop 1000s of rows

rows = 0
TARGET = 40000

while rows < TARGET:
    update_state()
    ax, ay, az, am = generate_accel()

    label = label_from_state()

    with open(DATASET_FILE, "a", newline="") as f:
        csv.writer(f).writerow([
            time.time(),
            round(emg, 3),
            round(temp, 3),
            round(sweat, 3),
            round(ax, 3),
            round(ay, 3),
            round(az, 3),
            round(am, 3),
            label
        ]) 
    rows += 1

    if rows % 1000 == 0:
        print(f"Generated {rows} rows")   

    time.sleep(0.02)
