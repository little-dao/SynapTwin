from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from collections import deque
import numpy as np
import pandas as pd
import datetime
import time
import google.generativeai as genai

# ========== CONFIG ==========
IP = "127.0.0.1"
PORT = 12345
SEGMENT_DURATION = 30  # in seconds (.5 minutes)
GEN_AI_MODEL = "gemini-1.5-pro-latest"  # Replace with the model you're using
MAX_BUFFER = 1800  # max points stored per signal (30 minutes at 6 Hz)

# Gemini API setup
genai.configure(api_key="AIzaSyBPOI74hfCYbDDrkSFUH-tTiJivcnndmxs")
model = genai.GenerativeModel(GEN_AI_MODEL)

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
    #print(f"{now.strftime('%H:%M:%S')} | {address}: {args}")

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
    eda = np.array(data_buffers["EDA"])
    ppg = np.array(data_buffers["PPG:IR"])
    temp = np.array(data_buffers["THERM"])
    if len(eda) == 0 or len(ppg) == 0 or len(temp) == 0:
        return None
    return {
        "eda_mean": float(np.mean(eda)),
        "eda_std": float(np.std(eda)),
        "hr_est": float(estimate_hr(ppg)),
        "temp_change": float(temp[-1] - temp[0]),
        "start_time": timestamp_buffer[0].strftime("%H:%M"),
        "end_time": timestamp_buffer[-1].strftime("%H:%M")
    }

# ========== AI SUMMARY ==========
def summarize_with_genai(features):
    prompt = f"""
This is biometric data from a wearable device between {features['start_time']} and {features['end_time']}:
- Avg EDA: {features['eda_mean']:.4f}
- EDA Variability: {features['eda_std']:.4f}
- Estimated Heart Rate: {features['hr_est']:.2f} bpm
- Skin Temperature Change: {features['temp_change']:.2f}°C

What can this say about the user's stress, focus, or general state? Respond like a friendly health assistant.
"""
    response = model.generate_content(prompt)
    return response.text.strip()

# ========== SEGMENT HANDLER ==========
def process_segment():
    features = extract_features()
    if features:
        summary = summarize_with_genai(features)
        log_entry = {
            "timestamp": f"{features['start_time']}–{features['end_time']}",
            "hr": features['hr_est'],
            "eda": features['eda_mean'],
            "temp_change": features['temp_change'],
            "summary": summary
        }
        segment_logs.append(log_entry)
        print("\n===== 🧠 Segment Summary =====")
        print(f"🕒 {log_entry['timestamp']}")
        print(summary)
        print("============================\n")
        save_summary_to_markdown(log_entry)

# ========== SAVE LOG ==========
def save_summary_to_markdown(entry):
    filename = f"biofeedback_journal_{datetime.date.today()}.md"
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"## 🕒 {entry['timestamp']}\n")
        f.write(f"**Heart Rate:** {entry['hr']:.2f} bpm\n")
        f.write(f"**EDA (avg):** {entry['eda']:.4f}\n")
        f.write(f"**Temp change:** {entry['temp_change']:.2f}°C\n")
        f.write(f"**Summary:** {entry['summary']}\n\n")


# ========== MAIN LOOP ==========
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
        if now - last_segment_time >= SEGMENT_DURATION:
            process_segment()
            last_segment_time = now

# ========== START ==========
if __name__ == "__main__":
    run_server()