import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("API Key not found.")
else:
    print(f"API Key found: {api_key[:5]}...")
    genai.configure(api_key=api_key)
    try:
        print("Listing available models...")
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print("-", m.name)
    except Exception as e:
        print("Error listing models:", e)
