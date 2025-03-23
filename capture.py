import time
import threading
from collections import deque
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from google import genai
from google.genai.types import HttpOptions
import os
from dotenv import load_dotenv

load_dotenv()


client = genai.Client(http_options=HttpOptions(api_version="v1"), api_key=os.getenv("GEMINI_API_KEY"))

# Store the last 5 seconds of messages
message_history = deque()
lock = threading.Lock()  # To prevent race conditions

def handle_all(address, *args):
    """Handles incoming OSC messages and stores them with timestamps."""
    global message_history
    timestamp = time.time()
    
    with lock:
        message_history.append((timestamp, address, args))
        
        # Remove messages older than 5 seconds
        while message_history and (timestamp - message_history[0][0] > 5):
            message_history.popleft()

def analyze_stress():
    """Runs every 5 seconds to analyze collected OSC data."""
    while True:
        time.sleep(5)
        
        with lock:
            if message_history:
                last_5_seconds_data = list(message_history)  # Copy the data
            
        if last_5_seconds_data:
            # Format the messages for Gemini
            formatted_data = "\n".join([f"{addr}: {args}" for _, addr, args in last_5_seconds_data])
            
            # Send to Gemini for stress analysis
            stress_report = analyze_with_gemini(formatted_data)
            print("\nStress Report:", stress_report)

def analyze_with_gemini(data):
    """Send data to Gemini API for analysis."""
    response = client.models.generate_content(
                model="gemini-2.0-flash-001",
                contents=f"You are a helpful AI companion designed to accompany people. You will monitor for signs of stress. :\n{data}")
    return response.text

# Setup OSC server
dispatcher = Dispatcher()
dispatcher.set_default_handler(handle_all)

ip = "127.0.0.1"
port = 12345
print(f"Listening on {ip}:{port}")

# Start the analysis thread
analysis_thread = threading.Thread(target=analyze_stress, daemon=True)
analysis_thread.start()

# Start the OSC server
server = BlockingOSCUDPServer((ip, port), dispatcher)
server.serve_forever()
