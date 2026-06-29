import ollama
import random
import time
import json
import os
import numpy as np

FEATURES = ["emg", "temp", "sweat", "accel_x", "accel_y", "accel_z", "accel_mag"]

# TensorFlow is optional: if it (or the trained model) is unavailable the demo
# degrades gracefully to the heuristic fallback_score(). Import it lazily here
# so a missing TensorFlow install does not crash the whole script.
try:
    from tensorflow.keras.models import load_model
    MODEL_AVAILABLE = True
except Exception:
    MODEL_AVAILABLE = False

MEMORY_FILE = "sensor_memory.json"
MODEL_FILE = "fatigue_bilstm.h5"

# LOAD MEMORY

if os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, "r") as f:
        try:
            memory = json.load(f)
        except:
            memory = []
else:
    memory = []

# LOAD MODEL

model = None

if MODEL_AVAILABLE and os.path.exists(MODEL_FILE):
    try:
        model = load_model(MODEL_FILE)
        print("BiLSTM model loaded.")
    except Exception as e:
        print("ERROR: Failed to load model:", e)

print("\n Athlete Cramp Risk AI v2\n")

# Sensor Input

def enrich_features(data):
    # Calculate acceleration magnitude
    accel_mag = (
        data["accel_x"]**2 + 
        data["accel_y"]**2 +
        data["accel_z"]**2
        ) ** 0.5
    
    return [
        data["emg"],
        data["temp"],
        data["sweat"],
        data["accel_x"],
        data["accel_y"],
        data["accel_z"],
        accel_mag
    ]

def get_sensor_data():

    # will be replaced with BLE later
    question = input("Enter 'sim' to simulate sensor data or 'manual' to input manually: ").strip().lower()
    
    if(question == "sim"):
        return {
            "emg": random.uniform(0,1),
            "temp": random.uniform(36,39),
            "sweat": random.uniform(0,1),
            "accel_x": random.uniform(-2,2),
            "accel_y": random.uniform(-2,2),
            "accel_z": random.uniform(-2,2)
        }
    elif (question == "manual"):
        try:
            emg = float(input("Enter EMG value (0-1): "))
            temp = float(input("Enter Temperature (36-39): "))
            sweat = float(input("Enter Sweat Level (0-1): "))
            accel_x = float(input("Enter Acceleration X (-2 to 2): "))
            accel_y = float(input("Enter Acceleration Y (-2 to 2): "))
            accel_z = float(input("Enter Acceleration Z (-2 to 2): "))
            return {
                "emg": emg,
                "temp": temp,
                "sweat": sweat,
                "accel_x": accel_x,
                "accel_y": accel_y,
                "accel_z": accel_z
            }
        except ValueError:
            print("Invalid input. Please enter numeric values.")
            return get_sensor_data()
sequence_buffer = []
SEQUENCE_LENGTH = 10 # num of recent sensor readings to consider for prediction

def fatigue_velocity(memory):
    if len(memory) < 5:
        return "Not enough data"
    
    recent = [x["score"] for x in memory[-5:]]

    return recent[-1] - recent[0]
def fallback_score(data):
    accel_mag = (
        data["accel_x"]**2 +
        data["accel_y"]**2 +
        data["accel_z"]**2
    ) ** 0.5
    
    # Calculate acceleration magnitude
    accel_mag = (data["accel_x"]**2 + data["accel_y"]**2 + data["accel_z"]**2)**0.5
    score = (
        data["emg"] * 40 +
        data["sweat"] * 25 +
        (data["temp"] -36) * 15 +
        min(accel_mag, 3) * 7
    )

    return max(0, min(100, round(score)))

# Basic BiLSTM prediction function

def predict_risk(data):

    global sequence_buffer

    features = enrich_features(data)

    sequence_buffer.append(features)

    if len(sequence_buffer) > SEQUENCE_LENGTH:
        sequence_buffer.pop(0)
    
    # lack of data, use fallback scoring
    if len(sequence_buffer) < SEQUENCE_LENGTH:
        return fallback_score(data)
    
    # prep input for BiLSTM

    if model is not None:
        try:
            X = np.array(sequence_buffer)
            X = X.reshape(
                1,
                SEQUENCE_LENGTH,
                len(FEATURES)
            )

            prediction = model.predict(
                X,
                verbose=0
            )[0][0]
            confidence = float(prediction)
            confidence = max(0.0, min(1.0, confidence)) 
            score = int(confidence * 100)

            if score < 5:
                score = 0
            return score

        
        except Exception as e:
            print("Prediction error:", e)
            return fallback_score(data)
        
    return fallback_score(data)

# analyze trend of L5

def calculate_trend(memory):

    if len(memory) < 10:
        return "Not enough data"
    
    scores = [
        x["score"]
        for x in memory[-10:]]


    latest = scores[-1]
    slope = latest - scores[0]
    avg = sum(scores) / len(scores)

    if slope > 10:
        return "Rapid risk increase"
    elif slope < -10:
        return "Rapid risk decrease"
    elif avg > 70:
        return "Sustained high risk"
    else:
        return "Stable risk"

# suggestion and recommendation engine

def recommendation(score):

    if score < 30:
        return (
            "Low risk. "
            "Continue training normally."
        )
    elif score < 60:
        return (
            "Moderate fatigue detected. "
            "Consider taking a short break, hydrating, and doing dynamic stretches. "
            "Monitor your recovery."
        )
    elif score < 80:
        return (
            "High fatigue detected. "
            "Stop training hydrate and perform static stretches. "
            "Consider a lighter recovery session for the next 24 hours."
        )
    else:
        return (
            "Very high risk of cramp or injury. "
            "Stop training / playing immediately, hydrate, and rest. "
            "Seek medical advice if symptoms persist or worsen."
        )

# Main Loop


while True:

    user = input(
        "\nPress Enter to simulate sensor data / manual / type exit to quit:"
    )

    if user.lower() == "exit":
        break

    if user.strip() == "":
        data = get_sensor_data()
    else:
        print("Invalid input. Just press Enter to simulate or type 'manual' to input data manually.")
        continue

    score = predict_risk(data)

    trend = calculate_trend(memory)

    advice = recommendation(score)

    print("\n=====================")
    print("Sensor Data Analysis")
    print("\n=====================")

    for key, value in data.items():
        print(f"{key}: {round(value,3)}")
    
    print("\nRisk Score:", score, "/100")
    print("Trend:", trend)
    print("Fatigue velocity:", fatigue_velocity(memory))
    print("AI Recommendation: ", advice)

    memory.append({
        "timestamp": time.time(),
        "data": data,
        "score": score
    })

    memory[:] = memory[-100:] # only keep last 100 entries

    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)
    
    prompt = f"""
You are a sports performance AI coach (designated to predict and prevent cramps and injurys).

Current snsor readings:

EMG: {data['emg']}
Temperature: {data['temp']}
Sweat: {data['sweat']}
Accel X: {data['accel_x']}
Accel Y: {data['accel_y']}
Accel Z: {data['accel_z']}

Risk Score:
{score}/100

Trend:
{trend}

Advice:
{advice}

Explain the athletes condition in a clear way,
possible causes of their score, risk ,
and what they can do to improve their condtion / performance.

Keep it motivating,
concise and professional,
and under 100 words.
"""

    try:
        response = ollama.chat(
            model="llama3",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        print("\nAI Coach Explanation")
        print("--------------------")
        print(
            response["message"]["content"]
        )

    except Exception as e:
        print("\nOllama API Error:",e)
    
    time.sleep(1) # wait a second
