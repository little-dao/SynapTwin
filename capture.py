import random

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from collections import deque
import numpy as np
import datetime
import time
import json
import os
import platform
from dotenv import load_dotenv
import google.generativeai as genai
import speech_recognition as sr
import pyttsx3


import asyncio
import edge_tts
import tempfile
import os
from playsound import playsound  # pip install playsound==1.2.2

# ===== LOAD ENV =====
load_dotenv()

# ===== CONFIG =====
IP = "127.0.0.1"
PORT = 12345
SEGMENT_DURATION = 3  # seconds
GEN_AI_MODEL = "gemini-2.0-flash-001"
MAX_BUFFER = 18000
JSON_LOG_PATH = "biometric_data.json"
CHAT_HISTORY_PATH = "chat_memory.json"
THRESHOLDS = None
Calibration_time=5
SEGBUFFER=100

# ===== Gemini Setup =====
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel(GEN_AI_MODEL)
chat = model.start_chat(history=[])

# ===== Buffers =====
data_buffers = {"EDA": deque(maxlen=MAX_BUFFER), "PPG:IR": deque(maxlen=MAX_BUFFER), "THERM": deque(maxlen=MAX_BUFFER)}
timestamp_buffer = deque(maxlen=MAX_BUFFER)
segment_logs = []
seg_buffers = {"EDA": deque(maxlen=SEGBUFFER), "PPG:IR": deque(maxlen=SEGBUFFER), "THERM": deque(maxlen=SEGBUFFER)}
timestamp_seg_buffer = deque(maxlen=SEGBUFFER)

# ===== Calibration Storage =====
sample_EDA_Mean=0
sample_EDA_Std=0
sample_PPG_Mean=0
sample_PPG_Std=0
sample_THERM_Mean=0
sample_THERM_Std=0



def speak(text):
    print(f"🔊 Assistant: {text}")

    async def _speak():
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
            output_path = tmpfile.name

        communicate = edge_tts.Communicate(text, voice="en-US-GuyNeural")
        await communicate.save(output_path)

        playsound(output_path)
        os.remove(output_path)

    asyncio.run(_speak())
# ===== Handlers =====
def handle_data(address, *args):
    now = datetime.datetime.now()
    timestamp_buffer.append(now)
    timestamp_seg_buffer.append(now)
    if "/EDA" in address:
        data_buffers["EDA"].append(args[0])
        seg_buffers["EDA"].append(args[0])
    elif "/PPG:IR" in address:
        data_buffers["PPG:IR"].append(args[0])
        seg_buffers["PPG:IR"].append(args[0])
    elif "/THERM" in address:
        data_buffers["THERM"].append(args[0])
        seg_buffers["THERM"].append(args[0])
#
def extract_features():
    eda = np.array(data_buffers["EDA"])
    ppg = np.array(data_buffers["PPG:IR"])
    temp = np.array(data_buffers["THERM"])
    if len(eda) == 0 or len(ppg) == 0 or len(temp) == 0:
        return None
    return {
        "eda_mean": float(np.mean(eda)),
        "eda_std": float(np.std(eda)),
        "ppg_mean": float(np.mean(ppg)),
        "ppg_std": float(np.std(ppg)),
        "temp_change": float(temp[-1] - temp[0]),
        "start_time": timestamp_buffer[0].strftime("%H:%M"),
        "end_time": timestamp_buffer[-1].strftime("%H:%M")
    }



def chat_with_user(features):
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    try:
        # 1. First message from Gemini
        prompt = f"""
        User now detected having higher EDA than usual with regard to 20s of biometric data. 
        Please initiate a gentle and empathetic conversation, checking how they're feeling and offering support.
        Your tone must feel human and caring. Make this a sentence or two.
        """+"User's current data from EMOTBIT:\n"+json.dumps(features)+"\n"+"User's historical EDA mean: "+str(sample_EDA_Mean)+"\n"+"User's historical EDA std: "+str(sample_EDA_Std)
        first_msg = chat.send_message(prompt).text
        speak(first_msg)

        # 2. Chat loop
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            while True:
                print("🎤 Listening for user input...")

                try:
                    audio = recognizer.listen(source, timeout=6, phrase_time_limit=50)
                    user_input = recognizer.recognize_google(audio)
                    print(f"🧑 User said: {user_input}")

                    # Check for exit signal
                    if any(word in user_input.lower() for word in ["i'm fine", "i'm okay", "quit", "thanks","thank you","bye bye"]):
                        speak("I'm glad you're feeling better. Take care, and I'm always here if you need to talk.")
                        break

                    # Gemini responds
                    follow_up_prompt = f"The user responded: \"{user_input}\". Continue the human-like, empathetic conversation. One sentence or two."
                    follow_up = chat.send_message(follow_up_prompt).text
                    speak(follow_up)

                except sr.UnknownValueError:
                    speak("Sorry, I didn't quite catch that. Could you try again?")
                except sr.WaitTimeoutError:
                    speak("Hmm, I didn't hear anything. Want to try again?")
                except sr.RequestError as e:
                    speak(f"Sorry, something went wrong with speech recognition. Error: {e}")
                    break

    except Exception as e:
        print(f"💥 Error in chat_with_user: {e}")
        speak("Something went wrong with the conversation. Let's try again later.")

def log_segment(features, summary=None, log_path="segment_data.json"):
    log_entry = {
        "timestamp": f"{features['start_time']}–{features['end_time']}",
        "eda_mean": features['eda_mean'],
        "eda_std": features['eda_std'],
        "ppg_mean": features['ppg_mean'],
        "ppg_std": features['ppg_std'],
        "temp_change": features['temp_change'],
        "summary": summary or "No summary"
    }
    segment_logs.append(log_entry)
    try:
        with open(log_path, "a") as f:
            json.dump(segment_logs, f, indent=2)
    except Exception as e:
        print(f"⚠️ Failed to write log: {e}")

def seg_features():
    eda=np.array(seg_buffers["EDA"])
    print(eda)
    ppg=np.array(seg_buffers["PPG:IR"])
    temp=np.array(seg_buffers["THERM"])
    if len(eda)==0 or len(ppg)==0 or len(temp)==0:
        return None
    return {
        "eda_mean": float(np.mean(eda)),
        "eda_std": float(np.std(eda)),
        "ppg_mean": float(np.mean(ppg)),
        "ppg_std": float(np.std(ppg)),
        "temp_change": float(temp[-1]-temp[0]),
        "start_time": timestamp_seg_buffer[0].strftime("%H:%M"),
        "end_time": timestamp_seg_buffer[-1].strftime("%H:%M")
    }

def process_segment():
    features = seg_features()
    if features:
        log_segment(features)
    if features:
        print(sample_EDA_Std)
        print(sample_EDA_Mean)
        print(features["eda_mean"])
        if (features["eda_mean"])>((sample_EDA_Mean+0.5)+sample_EDA_Std**2):
            print("EDA vol detected")
            for key in seg_buffers:
                seg_buffers[key].clear()
            timestamp_seg_buffer.clear()
            chat_with_user(features)
        #summary = summarize_with_genai(features)
        # segment_logs.append(log_entry)
        print("\n===== 🧠 Segment Summary =====")

def calibration():
    global sample_EDA_Mean, sample_EDA_Std, sample_PPG_Mean, sample_PPG_Std, sample_THERM_Mean
    features = extract_features()
    sample_EDA_Mean = features["eda_mean"]
    sample_EDA_Std = features["eda_std"]
    sample_PPG_Mean = features["ppg_mean"]
    sample_PPG_Std = features["ppg_std"]
    sample_THERM_Mean = features["temp_change"]
    print("Calibration complete. Monitoring now.")


# def summarize_with_genai():
#     try:
#         with open("segment_data.json") as f:
#             segments = json.load(f)
#         summary = chat.send_message(json.dumps(segments)).text
#         # sample = random.sample(segments, min(5, len(segments)))
#         prompt = (f"User's log data from EMOTBIT Data:\n{json.dumps(summary)}\n"+
#                   "User's historical EDA mean: "+str(sample_EDA_Mean)+"\n"+"User's historical EDA std: "+str(sample_EDA_Std)+"\n"+
#                   f"Summarize and analysis these data, respond with a human-like, empathetic message. Three or four sentences.")
#         response=chat.send_message(prompt).text
#         speak(response)
#         return None
#     except Exception as e:
#         print(f"⚠️ Failed to summarize with GenAI: {e}")
#         return None

def run_server():
    start=time.time()
    dispatcher = Dispatcher()
    dispatcher.map("/EmotiBit/0/EDA", handle_data)
    dispatcher.map("/EmotiBit/0/PPG:IR", handle_data)
    dispatcher.map("/EmotiBit/0/THERM", handle_data)

    server = BlockingOSCUDPServer((IP, PORT), dispatcher)
    print(f"✅ Listening on {IP}:{PORT}")
    server.handle_request()
    last = time.time()
    try:
        while time.time()-last<Calibration_time:
            server.handle_request()
        calibration()
        while True:
            server.handle_request()
            now = time.time()
            if now - last >= SEGMENT_DURATION:
                process_segment()
                last = now
            # if now - start >= 5:
            #     summarize_with_genai()

    except KeyboardInterrupt:
        print("🛑 Server exited.")

if __name__ == "__main__":
    run_server()