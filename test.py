import asyncio
import edge_tts
import tempfile
import os
from playsound import playsound  # pip install playsound==1.2.2

async def speak(text):
    print(f"🔊 Assistant: {text}")
    # Save speech to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
        output_path = tmpfile.name

    communicate = edge_tts.Communicate(text, voice="en-US-GuyNeural")
    await communicate.save(output_path)

    playsound(output_path)
    os.remove(output_path)  # Cleanup
asyncio.run(speak("Hey there! Just checking in. How are you feeling right now?"))
