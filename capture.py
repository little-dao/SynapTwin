from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
import numpy as np
import datetime
import time
import google.generativeai as genai
import speech_recognition as sr
import pyttsx3
import json
import os
from dotenv import load_dotenv
load_dotenv()

# Configuration
IP = "127.0.0.1"
PORT = 12345
SEGMENT_DURATION = 5  # in seconds
GEN_AI_MODEL = "gemini-2.0-flash-001"
THRESHOLDS = {"EDA": 0.5, "HR": 100, "TEMP": 0.5}  # Example thresholds
engine = pyttsx3.init()


genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel(GEN_AI_MODEL)

data_buffers = {"EDA": [], "HR": [], "TEMP": []}
timestamp_buffer = []

def handle_data(address, *args):
    now = datetime.datetime.now()
    timestamp_buffer.append(now)
    if "/EDA" in address:
        data_buffers["EDA"].append(args[0])
    elif "/PPG:IR" in address:
        data_buffers["HR"].append(args[0])
    elif "/THERM" in address:
        data_buffers["TEMP"].append(args[0])

# Analyze biometric data
def analyze_data():
    if len(data_buffers["EDA"]) == 0 or len(data_buffers["HR"]) == 0 or len(data_buffers["TEMP"]) == 0:
        return None
    
    eda_mean = np.mean(data_buffers["EDA"])
    hr_mean = np.mean(data_buffers["HR"])
    temp_change = data_buffers["TEMP"][-1] - data_buffers["TEMP"][0]
    
    stress_detected = eda_mean > THRESHOLDS["EDA"] or hr_mean > THRESHOLDS["HR"] or temp_change > THRESHOLDS["TEMP"]
    return stress_detected, eda_mean, hr_mean, temp_change

# Generate AI response
def chat_with_user():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    
    prompt = "Tell the user that they seem a bit stressed and if they want to talk about it."
    response = model.generate_content(prompt)
    speak(response.text)
    
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        try:
            print("Listening...")
            audio = recognizer.listen(source)
            user_input = recognizer.recognize_google(audio)
            response = model.generate_content(user_input)
            speak(response.text)
        except sr.UnknownValueError:
            speak("Sorry, I couldn't understand. Could you repeat that?")
        except sr.RequestError:
            speak("Error connecting to speech recognition service.")

def summarize_with_genai(features):
    prompt = f"""
    This is biometric data from a wearable device(EMOTIBIT) between {features['start_time']} and {features['end_time']}:
    - Avg EDA: {features['eda_mean']:.4f}
    - EDA Variability: {features['eda_std']:.4f}
    - Estimated Heart Rate: {features['hr_est']:.2f} bpm
    - Skin Temperature Change: {features['temp_change']:.2f}°C

    What can this say about the user's stress, focus, or general state? Respond like a friendly health assistant. Dont use markdown.
    """
    response = model.generate_content(prompt)
    return response.text.strip()

def speak(text):
    engine.say(text)
    engine.runAndWait()
    

def write_to_file(eda, hr, temp):
    data_entry = {
        "EDA": eda,
        "HR": hr,
        "TEMP": temp
    }
    
    file_path = "biometric_data.json"
    
    try:
        # Load existing data if the file exists
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
        else:
            data = []
        
        data.append(data_entry)
        
        # Write back to the file
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"Error writing to file: {e}")
    
def run_server():
    dispatcher = Dispatcher()
    dispatcher.map("/EmotiBit/0/EDA", handle_data)
    dispatcher.map("/EmotiBit/0/PPG:IR", handle_data)
    dispatcher.map("/EmotiBit/0/THERM", handle_data)
    
    server = BlockingOSCUDPServer((IP, PORT), dispatcher)
    print(f"✅ Listening on {IP}:{PORT}...")
    
    last_segment_time = time.time()
    while True:
        server.handle_request()
        now = time.time()
        write_to_file(data_buffers["EDA"], data_buffers["HR"], data_buffers["TEMP"])

        if now - last_segment_time >= SEGMENT_DURATION:
            stress_detected, eda, hr, temp = analyze_data()
            print(f"Stress Detected: {stress_detected}, eda: {eda}, hr: {hr}, temp: {temp}")
            if stress_detected:
                chat_with_user()
            else:
                features = {
                    "start_time": timestamp_buffer[0].strftime("%H:%M:%S"),
                    "end_time": timestamp_buffer[-1].strftime("%H:%M:%S"),
                    "eda_mean": eda,
                    "eda_std": np.std(data_buffers["EDA"]),
                    "hr_est": hr,
                    "temp_change": temp
                }
                response = summarize_with_genai(features)
                speak(response)
            last_segment_time = now
        
    

if __name__ == "__main__":
    run_server()