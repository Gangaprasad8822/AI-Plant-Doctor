import os
import requests
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# Test Weather
print("--- WEATHER API TEST ---")
weather_key = os.getenv("OPENWEATHER_API_KEY")
print(f"Weather Key: {weather_key}")
url = f"http://api.openweathermap.org/data/2.5/weather?q=Hyderabad&appid={weather_key}&units=metric"
r = requests.get(url)
print(f"Status Code: {r.status_code}")
print(f"Response: {r.text}")

# Test Gemini
print("\n--- GEMINI API TEST ---")
gemini_key = os.getenv("GEMINI_API_KEY")
print(f"Gemini Key: {gemini_key}")
try:
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content("hello")
    print(f"Gemini Response: {response.text}")
except Exception as e:
    import traceback
    traceback.print_exc()
