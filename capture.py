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
from scipy import signal
from sklearn.ensemble import IsolationForest

# ========== CONFIG ==========
IP = "127.0.0.1"
PORT = 12345
SEGMENT_DURATION = 5  # in seconds
GEN_AI_MODEL = "gemini-2.0-flash-001"
MAX_BUFFER = 1800  # max points stored per signal (30 minutes at 6 Hz)
JSON_LOG_PATH = "biometric_data.json"

# Baseline recording period (in seconds)
BASELINE_PERIOD = 5  # Record baseline for 1 minute

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

# Baseline values (will be calculated during initialization)
baseline = {
    "EDA_mean": None,
    "EDA_std": None,
    "HR_mean": None,
    "HR_std": None,
    "THERM_mean": None,
    "THERM_std": None
}

# Anomaly detection models
anomaly_models = {
    "EDA": None,
    "HR": None,
    "THERM": None
}

# Current stress level (0-100)
current_stress_level = 0

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
    """Estimate heart rate from PPG signal using peak detection"""
    if len(ppg_signal) < 30:
        return None
        
    # Normalize and detrend the signal
    normalized = np.array(ppg_signal)
    normalized = (normalized - np.mean(normalized)) / (np.std(normalized) if np.std(normalized) > 0 else 1)
    
    # Apply bandpass filter to isolate heart rate frequencies (0.5-3.5 Hz, typical HR 30-210 bpm)
    sos = signal.butter(4, [0.5, 3.5], 'bandpass', fs=30, output='sos')
    filtered = signal.sosfilt(sos, normalized)
    
    # Find peaks
    peaks, _ = signal.find_peaks(filtered, distance=8)  # minimum distance between peaks (assuming 30Hz)
    
    if len(peaks) < 2:
        return None
        
    # Calculate average time between peaks
    peak_intervals = np.diff(peaks)
    avg_interval = np.mean(peak_intervals)
    
    # Convert to BPM (assuming 30Hz sampling rate)
    hr_bpm = 60 / (avg_interval / 30)
    
    # Sanity check - typical human HR range
    if 40 <= hr_bpm <= 200:
        return hr_bpm
    else:
        return None

def extract_hrv(ppg_signal):
    """Extract heart rate variability from PPG signal"""
    if len(ppg_signal) < 60:
        return None
        
    # Normalize and detrend
    normalized = np.array(ppg_signal)
    normalized = (normalized - np.mean(normalized)) / (np.std(normalized) if np.std(normalized) > 0 else 1)
    
    # Apply bandpass filter
    sos = signal.butter(4, [0.5, 3.5], 'bandpass', fs=30, output='sos')
    filtered = signal.sosfilt(sos, normalized)
    
    # Find peaks (R-peaks equivalent in PPG)
    peaks, _ = signal.find_peaks(filtered, distance=8)
    
    if len(peaks) < 3:
        return None
        
    # Calculate NN intervals (time between peaks)
    intervals = np.diff(peaks) / 30  # convert to seconds
    
    # RMSSD (Root Mean Square of Successive Differences)
    successive_diffs = np.diff(intervals)
    rmssd = np.sqrt(np.mean(successive_diffs**2)) * 1000  # convert to ms
    
    # SDNN (Standard Deviation of NN intervals)
    sdnn = np.std(intervals) * 1000  # convert to ms
    
    return {
        "rmssd": rmssd,
        "sdnn": sdnn
    }

def extract_scr_features(eda_signal):
    """Extract Skin Conductance Response features from EDA signal"""
    if len(eda_signal) < 30:
        return None
        
    # Convert to numpy array
    eda = np.array(eda_signal)
    
    # Split into tonic (SCL) and phasic (SCR) components using low-pass filter
    sos = signal.butter(4, 0.05, 'lowpass', fs=30, output='sos')
    scl = signal.sosfilt(sos, eda)  # Skin Conductance Level (tonic)
    scr = eda - scl  # Skin Conductance Response (phasic)
    
    # Find SCR peaks
    peaks, _ = signal.find_peaks(scr, height=0.01, distance=30)  # minimum 1 second between peaks
    
    return {
        "scl_mean": np.mean(scl),
        "scr_amplitude": np.max(scr) if len(scr) > 0 else 0,
        "scr_count": len(peaks),
        "scr_rise_time": np.mean(np.diff(peaks)) / 30 if len(peaks) > 1 else 0
    }

def extract_features():
    """Extract comprehensive features from all signals"""
    eda = list(data_buffers["EDA"])
    ppg = list(data_buffers["PPG:IR"])
    temp = list(data_buffers["THERM"])
    
    if len(eda) < 30 or len(ppg) < 30 or len(temp) < 2:
        return None
    
    # Basic statistics
    hr_est = estimate_hr(ppg)
    hrv = extract_hrv(ppg)
    scr_features = extract_scr_features(eda)
    
    # Handle None values
    if hr_est is None:
        hr_est = 75.0  # fallback to average
    
    if hrv is None:
        hrv = {"rmssd": 30.0, "sdnn": 50.0}  # fallback to average
        
    if scr_features is None:
        scr_features = {
            "scl_mean": np.mean(eda),
            "scr_amplitude": 0,
            "scr_count": 0,
            "scr_rise_time": 0
        }
    
    features = {
        # Time information
        "start_time": timestamp_buffer[0].strftime("%H:%M:%S") if len(timestamp_buffer) > 0 else "00:00:00",
        "end_time": timestamp_buffer[-1].strftime("%H:%M:%S") if len(timestamp_buffer) > 0 else "00:00:00",
        "timestamp": datetime.datetime.now().isoformat(),
        
        # Heart rate features
        "hr": float(hr_est),
        "rmssd": float(hrv["rmssd"]),
        "sdnn": float(hrv["sdnn"]),
        
        # EDA features
        "eda_mean": float(np.mean(eda)),
        "eda_std": float(np.std(eda)),
        "scl": float(scr_features["scl_mean"]),
        "scr_amplitude": float(scr_features["scr_amplitude"]),
        "scr_count": int(scr_features["scr_count"]),
        "scr_rise_time": float(scr_features["scr_rise_time"]),
        
        # Temperature features
        "temp": float(np.mean(temp)),
        "temp_change": float(temp[-1] - temp[0])
    }
    
    return features

def speak(text):
    """Function to convert text to speech"""
    print(f"🔊 Assistant: {text}")
    engine.say(text)
    engine.runAndWait()

def listen_to_user():
    """Listen to user's speech input"""
    recognizer = sr.Recognizer()
    
    try:
        with sr.Microphone() as source:
            print("Listening...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=15)
            
            user_input = recognizer.recognize_google(audio)
            print(f"👤 User: {user_input}")
            return user_input
    except sr.UnknownValueError:
        print("Could not understand audio")
        return None
    except sr.RequestError:
        print("Could not request results")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def chat_with_user_until_calm(features):
    """Conducts an ongoing conversation until stress levels decrease"""
    conversation_log = []
    stress_level = calculate_stress_level(features)
    stress_history = [stress_level]
    
    # Initial conversation context
    context = f"""
    The user is showing elevated stress levels:
    - Heart rate: {features['hr']:.1f} bpm
    - EDA reading: {features['eda_mean']:.2f}
    - Temperature: {features['temp']:.2f}°C
    
    You are having a conversation to help them reduce their stress. Keep responses supportive,
    brief, and focused on guiding them to a calmer state. Ask open-ended questions about their
    current feelings and offer simple relaxation techniques when appropriate.
    """
    
    # Initial greeting
    prompt = context + "\nGenerate a brief, empathetic greeting, asking how they're feeling right now."
    response = model.generate_content(prompt)
    speak(response.text)
    conversation_log.append({"role": "assistant", "content": response.text})
    
    # Continue conversation until stress decreases
    turns = 0
    max_turns = 2  # Maximum conversation turns before checking stress again
    
    while turns < max_turns:
        # Get user input
        user_input = listen_to_user()
        if user_input is None:
            speak("I didn't catch that. Let's try again.")
            continue
            
        conversation_log.append({"role": "user", "content": user_input})
        
        # Generate response based on conversation history
        conversation_history = "\n".join([f"{'User' if entry['role'] == 'user' else 'Assistant'}: {entry['content']}" 
                                        for entry in conversation_log[-4:]])  # Last 4 exchanges
        
        prompt = f"""
        {context}
        
        Conversation history:
        {conversation_history}
        
        Based on this conversation, generate a supportive response that helps reduce their stress.
        Keep it brief (1-2 sentences) and either:
        1. Acknowledge their feelings and offer a simple coping strategy, or
        2. Ask a question that guides them toward identifying what would help them feel better.
        """
        
        response = model.generate_content(prompt)
        speak(response.text)
        conversation_log.append({"role": "assistant", "content": response.text})
        
        turns += 1
        
        # Every 2 turns, check if stress has decreased
        if turns % 2 == 0:
            # Re-extract features and calculate stress
            new_features = extract_features()
            if new_features:
                new_stress = calculate_stress_level(new_features)
                stress_history.append(new_stress)
                
                # If stress has decreased significantly, prepare to end conversation
                if new_stress < stress_level * 0.7:  # 30% reduction
                    prompt = f"""
                    The user's stress levels have decreased. Generate a brief, positive closing message
                    acknowledging their progress and offering one simple reminder for staying calm.
                    """
                    response = model.generate_content(prompt)
                    speak(response.text)
                    conversation_log.append({"role": "assistant", "content": response.text})
                    break
    
    # Final assessment
    final_features = extract_features()
    final_stress = calculate_stress_level(final_features) if final_features else stress_level
    stress_history.append(final_stress)
    
    return {
        "conversation": conversation_log,
        "stress_trajectory": stress_history,
        "stress_reduced": final_stress < stress_level
    }

def establish_baseline():
    """Establish baseline readings for the individual"""
    print("⏱️ Establishing baseline readings for 60 seconds...")
    speak("I'm calibrating to your normal levels. Please relax for 20 seconds.")
    
    start_time = time.time()
    baseline_data = {
        "EDA": [],
        "HR": [],
        "THERM": []
    }
    
    # Collect baseline data
    while time.time() - start_time < BASELINE_PERIOD:
        time.sleep(0.5)  # Check every half second
        
        # Extract current features
        features = extract_features()
        if features:
            baseline_data["EDA"].append(features["eda_mean"])
            baseline_data["HR"].append(features["hr"])
            baseline_data["THERM"].append(features["temp"])
    
    # Calculate baseline statistics
    if len(baseline_data["EDA"]) > 0:
        baseline["EDA_mean"] = np.mean(baseline_data["EDA"])
        baseline["EDA_std"] = np.std(baseline_data["EDA"]) if len(baseline_data["EDA"]) > 1 else 0.1
        
        baseline["HR_mean"] = np.mean(baseline_data["HR"])
        baseline["HR_std"] = np.std(baseline_data["HR"]) if len(baseline_data["HR"]) > 1 else 5
        
        baseline["THERM_mean"] = np.mean(baseline_data["THERM"])
        baseline["THERM_std"] = np.std(baseline_data["THERM"]) if len(baseline_data["THERM"]) > 1 else 0.1
        
        print(f"✅ Baseline established:")
        print(f"   - EDA: {baseline['EDA_mean']:.2f} ± {baseline['EDA_std']:.2f}")
        print(f"   - HR: {baseline['HR_mean']:.1f} ± {baseline['HR_std']:.1f} bpm")
        print(f"   - Temperature: {baseline['THERM_mean']:.2f} ± {baseline['THERM_std']:.2f}°C")
        
        # Train anomaly detection models with baseline data
        for signal in ["EDA", "HR", "THERM"]:
            if len(baseline_data[signal]) > 10:  # Need enough samples
                # Reshape for isolation forest
                X = np.array(baseline_data[signal]).reshape(-1, 1)
                # Train model with contamination=0.1 (assuming 10% of baseline might be abnormal)
                anomaly_models[signal] = IsolationForest(contamination=0.1, random_state=42).fit(X)
        
        speak("Baseline readings established. I'm now monitoring your biometrics.")
        return True
    else:
        print("❌ Could not establish baseline. Not enough data.")
        speak("I couldn't establish baseline readings. Please check the sensor connections.")
        return False

def calculate_stress_level(features):
    """Calculate a stress level score (0-100) based on multiple factors"""
    if not all(v is not None for v in baseline.values()):
        # If baseline not established, use simple threshold approach
        stress_score = 0
        if features["hr"] > 90:
            stress_score += 30
        if features["eda_mean"] > 3.0:
            stress_score += 30
        if abs(features["temp_change"]) > 1.0:
            stress_score += 20
        if features["scr_count"] > 3:
            stress_score += 20
        return min(100, stress_score)
    
    # Calculate z-scores (deviation from baseline)
    hr_z = (features["hr"] - baseline["HR_mean"]) / baseline["HR_std"] if baseline["HR_std"] > 0 else 0
    eda_z = (features["eda_mean"] - baseline["EDA_mean"]) / baseline["EDA_std"] if baseline["EDA_std"] > 0 else 0
    temp_z = (features["temp"] - baseline["THERM_mean"]) / baseline["THERM_std"] if baseline["THERM_std"] > 0 else 0
    
    # Use anomaly detection models if available
    anomaly_scores = []
    for signal, model in anomaly_models.items():
        if model is not None:
            if signal == "EDA":
                value = features["eda_mean"]
            elif signal == "HR":
                value = features["hr"]
            elif signal == "THERM":
                value = features["temp"]
                
            # Get anomaly score (-1 for anomalies, 1 for normal)
            score = model.score_samples(np.array([[value]]))[0]
            # Convert to 0-1 scale where higher means more anomalous
            normalized_score = (1 - (score + 1) / 2)
            anomaly_scores.append(normalized_score)
    
    # Combine all factors into a stress score
    stress_components = []
    
    # 1. Heart rate variability (lower RMSSD = higher stress)
    if features["rmssd"] < 20:  # Low HRV indicates stress
        hrv_stress = (20 - features["rmssd"]) / 20 * 25  # Max 25 points
        stress_components.append(("HRV", min(25, hrv_stress)))
    
    # 2. Elevated heart rate
    if hr_z > 0:  # Only count elevated HR, not lower
        hr_stress = min(25, hr_z * 10)  # Max 25 points
        stress_components.append(("HR", hr_stress))
    
    # 3. Elevated EDA
    if eda_z > 0:  # Only count elevated EDA
        eda_stress = min(20, eda_z * 8)  # Max 20 points
        stress_components.append(("EDA", eda_stress))
    
    # 4. SCR components
    scr_stress = min(15, features["scr_count"] * 3 + features["scr_amplitude"] * 10)
    stress_components.append(("SCR", scr_stress))
    
    # 5. Temperature change
    temp_stress = min(10, abs(temp_z) * 5)  # Max 10 points
    stress_components.append(("TEMP", temp_stress))
    
    # 6. Anomaly detection
    if anomaly_scores:
        anomaly_stress = min(20, np.mean(anomaly_scores) * 30)  # Max 20 points
        stress_components.append(("ANOMALY", anomaly_stress))
    
    # Calculate total stress score
    stress_score = sum(score for _, score in stress_components)
    stress_score = min(100, max(0, stress_score))  # Clamp to 0-100
    
    return stress_score

def detect_stress(features):
    """Determine if the user is stressed based on features"""
    # Calculate comprehensive stress score
    stress_score = calculate_stress_level(features)
    
    # Update global stress level
    global current_stress_level
    current_stress_level = stress_score
    
    # Consider stressed if score exceeds threshold
    # 50+ indicates moderate stress, 70+ indicates high stress
    return stress_score >= 50, stress_score

def summarize_with_genai(features, stress_score):
    prompt = f"""
    Biometric data from {features['start_time']} to {features['end_time']}:
    - Heart rate: {features['hr']:.1f} bpm (HRV RMSSD: {features['rmssd']:.1f} ms)
    - EDA: {features['eda_mean']:.4f} (SCR count: {features['scr_count']})
    - Temperature: {features['temp']:.2f}°C (change: {features['temp_change']:.2f}°C)
    - Overall stress score: {stress_score}/100
    
    As a health assistant, provide a brief assessment of their current physiological state.
    Mention if values indicate relaxation or potential stress. Give one practical suggestion
    appropriate to their current state. Keep your response under 3 sentences.
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

segment_emotions = []
segment_times = []

def analyze_emotions(features):
    """Analyze features to infer likely emotional states"""
    if features is None:
        return "Unknown"
    
    emotions = []
    
    # Use HR and EDA to infer emotional states
    hr = features["hr"]
    eda_mean = features["eda_mean"]
    scr_count = features["scr_count"]
    temp = features["temp"]
    
    # Simple rule-based emotion inference
    # Note: This is a simplified model - real emotion detection requires more sophisticated algorithms
    if hr > 90 and eda_mean > 5:
        if scr_count > 3:
            emotions.append("Excited")
        else:
            emotions.append("Stressed")
    elif hr > 85 and eda_mean > 3:
        emotions.append("Engaged")
    elif hr < 70 and eda_mean < 2:
        emotions.append("Relaxed")
    elif hr < 65 and temp < 36:
        emotions.append("Tired")
    else:
        emotions.append("Neutral")
    
    # Additional emotional nuances based on HRV
    if features["rmssd"] < 20:
        emotions.append("Tense")
    elif features["rmssd"] > 50:
        emotions.append("Calm")
    
    return emotions

def generate_emotion_timeline(segment_emotions, segment_times):
    """Generate a text-based timeline of emotions"""
    if not segment_emotions or not segment_times:
        return "No emotional data available yet"
    
    timeline = "📊 EMOTION TIMELINE:\n"
    timeline += "═══════════════════════════════\n"
    
    for i, (time, emotions) in enumerate(zip(segment_times, segment_emotions)):
        start_time = time.strftime("%H:%M:%S")
        emotion_str = ", ".join(emotions) if emotions else "Unknown"
        
        # Use emoji indicators for emotional states
        emoji = "😐"  # default
        if "Excited" in emotions: emoji = "😃"
        elif "Stressed" in emotions: emoji = "😰"
        elif "Engaged" in emotions: emoji = "🤔"
        elif "Relaxed" in emotions: emoji = "😌"
        elif "Tired" in emotions: emoji = "😴"
        elif "Tense" in emotions: emoji = "😬"
        elif "Calm" in emotions: emoji = "😊"
        
        timeline += f"{start_time} {emoji} {emotion_str}\n"
    
    timeline += "═══════════════════════════════\n"
    return timeline

def process_segment():
    """Process a segment of data and take appropriate action"""
    features = extract_features()
    if not features:
        print("⚠️ Not enough data to analyze yet")
        return
    
    emotions = analyze_emotions(features)
    segment_emotions.append(emotions)
    segment_times.append(datetime.datetime.now())
    
    # Check stress using improved method
    is_stressed, stress_score = detect_stress(features)
    
    log_entry = {
        "timestamp": features["timestamp"],
        "time_range": f"{features['start_time']}–{features['end_time']}",
        "biometrics": {
            "hr": features['hr'],
            "rmssd": features['rmssd'],
            "sdnn": features['sdnn'],
            "eda": features['eda_mean'],
            "eda_std": features['eda_std'],
            "scr_count": features['scr_count'],
            "scr_amplitude": features['scr_amplitude'],
            "temp": features['temp'],
            "temp_change": features['temp_change']
        },
        "stress_score": stress_score,
        "is_stressed": is_stressed
    }
    
    # If stress detected, initiate conversation until calm
    if is_stressed:
        print(f"\n🚨 STRESS DETECTED - Stress Score: {stress_score:.1f}/100")
        print("Starting conversation to help reduce stress...")
        
        # Continue conversation until stress levels decrease
        conversation_result = chat_with_user_until_calm(features)
        
        log_entry["conversation"] = conversation_result["conversation"]
        log_entry["stress_reduced"] = conversation_result["stress_reduced"]
        log_entry["stress_trajectory"] = conversation_result["stress_trajectory"]
        
        # Get final features after conversation
        final_features = extract_features()
        if final_features:
            _, final_stress = detect_stress(final_features)
            log_entry["final_stress_score"] = final_stress
            
            print(f"Conversation complete. Stress level: {stress_score:.1f} → {final_stress:.1f}")
            if final_stress < stress_score:
                print("✅ Successfully reduced stress levels")
            else:
                print("⚠️ Stress levels remained elevated")
        
        log_entry["summary"] = f"Stress intervention conducted. Initial score: {stress_score:.1f}/100."
        
        if len(segment_emotions) > 0:
            timeline = generate_emotion_timeline(segment_emotions, segment_times)
            print("\n" + timeline + "\n")
    else:
        # Otherwise just get a summary
        summary = summarize_with_genai(features, stress_score)
        log_entry["summary"] = summary
        
        print("\n===== 🧠 Segment Summary =====")
        print(f"🕒 {log_entry['time_range']}")
        print(f"💓 HR: {features['hr']:.1f} bpm | 💧 EDA: {features['eda_mean']:.4f}")
        print(f"🧘 Stress Score: {stress_score:.1f}/100")
        print(summary)
        print("============================\n")
        
        if len(segment_emotions) > 0:
            timeline = generate_emotion_timeline(segment_emotions, segment_times)
            print("\n" + timeline + "\n")
    
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
    print(f"📂 Logging data to {JSON_LOG_PATH}")
    print(f"⏱️ Processing segments every {SEGMENT_DURATION} seconds")
    
    # Wait for initial data collection
    time.sleep(5)
    print("Initial data collection complete")
    
    # Establish baseline
    baseline_established = establish_baseline()
    if not baseline_established:
        print("⚠️ Continuing without baseline. Will use default thresholds.")
    
    print("🚀 Monitoring started. Press Ctrl+C to exit")
    
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