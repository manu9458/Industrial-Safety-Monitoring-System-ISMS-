import os
import cv2
import base64
import pyttsx3
import threading
import time
from dotenv import load_dotenv
from openai import OpenAI

# Load API Key
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    print("Error: OPENAI_API_KEY not found in .env file.")

# Configure OpenAI
try:
    client = OpenAI(api_key=API_KEY)
except Exception as e:
    print(f"OpenAI Config Error: {e}")
    client = None

# Initialize TTS Engine
engine = pyttsx3.init()
engine.setProperty('rate', 150) # Speed of speech

class SmartAssistant:
    def __init__(self):
        self.is_processing = False
        self.last_analysis_time = 0
        self.cooldown = 10 # Seconds between AI voice warnings

    def speak(self, text):
        """Threaded text-to-speech."""
        def _speak_task():
            try:
                local_engine = pyttsx3.init() 
                local_engine.say(text)
                local_engine.runAndWait()
            except Exception as e:
                print(f"TTS Error: {e}")

        t = threading.Thread(target=_speak_task)
        t.start()

    def encode_image(self, frame):
        """Encodes OpenCV frame to base64 string."""
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')

    def analyze_scene(self, frame, trigger_reason="Routine Check"):
        if self.is_processing:
            return
        
        if time.time() - self.last_analysis_time < self.cooldown:
            return

        self.last_analysis_time = time.time()
        self.is_processing = True

        def _ai_task():
            if client is None:
                print("Error: OpenAI Client not initialized. Check API Key.")
                return

            try:
                print(f"🤖 Frame sent to OpenAI... Reason: {trigger_reason}")
                
                # Encode image
                base64_image = self.encode_image(frame)

                # Prompt Engineering
                prompt = (
                    f"You are an Industrial Safety AI. Context: {trigger_reason}. "
                    "Analyze the image and find the specific person violating safety rules. "
                    "Describe their visual appearance (shirt color, location) to identify them. "
                    "Generate a spoken warning exactly like this: "
                    "'Attention! The worker in the [Color] shirt, you are missing your hard hat. Please equip it immediately.' "
                    "If there are multiple risks, mention them briefly."
                    "If everything is safe, just reply 'SAFE'."
                )

                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    },
                                },
                            ],
                        }
                    ],
                    max_tokens=300
                )

                text = response.choices[0].message.content.strip()
                
                print(f"🤖 OpenAI says: {text}")

                if "SAFE" not in text.upper():
                    self.speak(text)
                
            except Exception as e:
                print(f"AI Analysis Error: {e}")
            finally:
                self.is_processing = False

        t = threading.Thread(target=_ai_task)
        t.start()
