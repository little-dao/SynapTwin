from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from collections import deque
import numpy as np
import pandas as pd
import datetime
import time
import json
import os
import google.generativeai as genai
import speech_recognition as sr
import pyttsx3

# ========== CONFIG ==========
IP = "127.0.0.1"
PORT = 12345
SEGMENT_DURATION = 5  # in seconds (.5 minutes)
GEN_AI_MODEL = "gemini-2.0-flash-001"
MAX_BUFFER = 1800  # max points stored per signal (30 minutes at 6 Hz)
JSON_LOG_PATH = "biometric_data.json"

# Stress thresholds - adjust these based on your specific sensor calibrations
THRESHOLDS = {
    "EDA": 3.5,       # Electrodermal activity threshold (microsiemens)
    "HR": 90,         # Heart rate threshold (bpm)
    "TEMP_CHANGE": 1.0  # Temperature change threshold (°C)
}

# Gemini API setup
genai.configure(api_key="AIzaSyBPOI74hfCYbDDrkSFUH-tTiJivcnndmxs")
model = genai.GenerativeModel(GEN_AI_MODEL)

# Speech synthesis setup
engine = pyttsx3.init()
engine.setProperty('rate', 180)  # Speed of speech
engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)

# Data Buffers
data_buffers = {
    "EDA": deque(maxlen=MAX_BUFFER),
    "PPG:IR": deque(maxlen=MAX_BUFFER),
    "THERM": deque(maxlen=MAX_BUFFER)
}
timestamp_buffer = deque(maxlen=MAX_BUFFER)

# Segment logs
segment_logs = []

# ========== DATA HANDLER ==========
def handle_data(address, *args):
    now = datetime.datetime.now()
    timestamp_buffer.append(now)
    if "/EDA" in address:
        data_buffers["EDA"].append(args[0])
    elif "/PPG:IR" in address:
        data_buffers["PPG:IR"].append(args[0])
    elif "/THERM" in address:
        data_buffers["THERM"].append(args[0])

# ========== FEATURE EXTRACTION ==========
def estimate_hr(ppg_signal):
    # Placeholder: assume 30Hz sample rate and find dominant freq
    if len(ppg_signal) < 10:
        return None
    detrended = np.array(ppg_signal) - np.mean(ppg_signal)
    freqs = np.fft.rfftfreq(len(detrended), d=1/30)
    fft = np.abs(np.fft.rfft(detrended))
    dominant_freq = freqs[np.argmax(fft[1:]) + 1]  # skip DC component
    return dominant_freq * 60  # convert Hz to bpm

def extract_features():
    eda = list(data_buffers["EDA"])
    ppg = list(data_buffers["PPG:IR"])
    temp = list(data_buffers["THERM"])
    
    if len(eda) == 0 or len(ppg) == 0 or len(temp) == 0:
        return None
    
    hr_est = estimate_hr(ppg)
    if hr_est is None:
        hr_est = 0.0
        
    return {
        "eda_mean": float(np.mean(eda)),
        "eda_std": float(np.std(eda)),
        "hr_est": float(hr_est),
        "temp_change": float(temp[-1] - temp[0]) if len(temp) > 1 else 0.0,
        "start_time": timestamp_buffer[0].strftime("%H:%M:%S") if len(timestamp_buffer) > 0 else "00:00:00",
        "end_time": timestamp_buffer[-1].strftime("%H:%M:%S") if len(timestamp_buffer) > 0 else "00:00:00",
        "timestamp": datetime.datetime.now().isoformat()
    }

def speak(text):
    """Function to convert text to speech"""
    print(f"🔊 Assistant: {text}")
    engine.say(text)
    engine.runAndWait()

def chat_with_user(features):
    """Initiates a conversation with the user when stress is detected"""
    recognizer = sr.Recognizer()
    
    try:
        with sr.Microphone() as source:
            # Generate initial prompt based on the biometric data
            stress_prompt = f"""
            The user's biometrics show possible stress:
            - Heart rate: {features['hr_est']:.1f} bpm
            - EDA reading: {features['eda_mean']:.2f}
            - Temperature change: {features['temp_change']:.2f}°C
            
            Generate a brief, empathetic message checking in with them about their stress levels. 
            Keep it under 20 words and ask an open-ended question.
            """
            
            response = model.generate_content(stress_prompt)
            speak(response.text)
            
            # Listen for user response
            print("Listening...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            
            user_input = recognizer.recognize_google(audio)
            print(f"👤 User: {user_input}")
            
            # Generate response to user
            follow_up_prompt = f"""
            The user responded with: "{user_input}"
            
            Generate a supportive, brief response offering one practical suggestion to help them manage stress.
            Keep it under 30 words.
            """
            
            follow_up = model.generate_content(follow_up_prompt)
            speak(follow_up.text)
            
            # Return the conversation for logging
            return {
                "assistant_first": response.text,
                "user_response": user_input,
                "assistant_second": follow_up.text
            }
            
    except sr.UnknownValueError:
        speak("I'm sorry, I couldn't understand what you said.")
        return {"error": "Speech not recognized"}
    except sr.RequestError:
        speak("Sorry, I couldn't connect to the speech recognition service.")
        return {"error": "Speech service connection failed"}
    except Exception as e:
        speak("I'm having trouble with our conversation right now.")
        return {"error": str(e)}

def check_stress_levels(features):
    """Checks if stress thresholds are exceeded"""
    if features is None:
        return False
        
    stress_detected = (
        features['eda_mean'] > THRESHOLDS["EDA"] or
        features['hr_est'] > THRESHOLDS["HR"] or
        features['temp_change'] > THRESHOLDS["TEMP_CHANGE"]
    )
    
    return stress_detected

def summarize_with_genai(features):
    prompt = f"""
    This is biometric data from a wearable device between {features['start_time']} and {features['end_time']}:
    - Avg EDA: {features['eda_mean']:.4f}
    - EDA Variability: {features['eda_std']:.4f}
    - Estimated Heart Rate: {features['hr_est']:.2f} bpm
    - Skin Temperature Change: {features['temp_change']:.2f}°C

    What can this say about the user's stress, focus, or general state? Respond like a friendly health assistant.
    Keep it brief and actionable, under 3 sentences.
    """
    response = model.generate_content(prompt)
    return response.text.strip()

def save_to_json(data):
    """Save data to JSON file"""
    try:
        # Load existing data if file exists
        if os.path.exists(JSON_LOG_PATH):
            with open(JSON_LOG_PATH, 'r') as file:
                try:
                    existing_data = json.load(file)
                except json.JSONDecodeError:
                    existing_data = []
        else:
            existing_data = []
        
        # Append new data
        existing_data.append(data)
        
        # Write back to file
        with open(JSON_LOG_PATH, 'w') as file:
            json.dump(existing_data, file, indent=2)
            
        print(f"✅ Data logged to {JSON_LOG_PATH}")
    except Exception as e:
        print(f"❌ Error saving to JSON: {e}")

def process_segment():
    features = extract_features()
    if not features:
        print("⚠️ Not enough data to analyze yet")
        return
        
    # Check if stress thresholds are exceeded
    stress_detected = check_stress_levels(features)
    
    # Create log entry
    log_entry = {
        "timestamp": features["timestamp"],
        "time_range": f"{features['start_time']}–{features['end_time']}",
        "hr": features['hr_est'],
        "eda": features['eda_mean'],
        "eda_std": features['eda_std'],
        "temp_change": features['temp_change'],
        "stress_detected": stress_detected
    }
    
    # If stress detected, initiate conversation
    if stress_detected:
        print("\n🚨 STRESS DETECTED - INITIATING CONVERSATION")
        conversation = chat_with_user(features)
        log_entry["conversation"] = conversation
        log_entry["summary"] = "Conversation initiated due to elevated stress levels."
    else:
        # Otherwise just get a summary
        summary = summarize_with_genai(features)
        log_entry["summary"] = summary
        print("\n===== 🧠 Segment Summary =====")
        print(f"🕒 {log_entry['time_range']}")
        print(f"💓 HR: {log_entry['hr']:.1f} bpm | 💧 EDA: {log_entry['eda']:.4f} | 🌡️ Temp Change: {log_entry['temp_change']:.2f}°C")
        print(f"⚠️ Stress Detected: No")
        print(summary)
        print("============================\n")
    
    # Save to segment logs and JSON
    segment_logs.append(log_entry)
    save_to_json(log_entry)

# ========== MAIN LOOP ==========
def run_server():
    dispatcher = Dispatcher()
    dispatcher.map("/EmotiBit/0/EDA", handle_data)
    dispatcher.map("/EmotiBit/0/PPG:IR", handle_data)
    dispatcher.map("/EmotiBit/0/THERM", handle_data)

    server = BlockingOSCUDPServer((IP, PORT), dispatcher)
    print(f"✅ Listening on {IP}:{PORT}...")
    print(f"⚠️ Stress thresholds: EDA > {THRESHOLDS['EDA']}, HR > {THRESHOLDS['HR']} bpm, Temp change > {THRESHOLDS['TEMP_CHANGE']}°C")
    print(f"📂 Logging data to {JSON_LOG_PATH}")
    print(f"⏱️ Processing segments every {SEGMENT_DURATION} seconds")
    print("Press Ctrl+C to exit")

    last_segment_time = time.time()
    try:
        while True:
            # This makes the server non-blocking, processes one request then continues
            server.handle_request()
            
            now = time.time()
            if now - last_segment_time >= SEGMENT_DURATION:
                process_segment()
                last_segment_time = now
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    finally:
        print(f"💾 Data saved to {JSON_LOG_PATH}")

if __name__ == "__main__":
    run_server()