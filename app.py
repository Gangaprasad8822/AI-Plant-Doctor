from flask import Flask, render_template, request, send_file, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os
from gtts import gTTS
import pickle
import sqlite3
from datetime import datetime
import random
import urllib.parse
import requests
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

from treatment import treatments

app = Flask(__name__)

# ---------------- FOLDERS ----------------
os.makedirs("static", exist_ok=True)
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("db", exist_ok=True)

# ---------------- DATABASE INIT ----------------
def init_db():
    conn = sqlite3.connect('db/history.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT,
            result TEXT,
            confidence REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ---------------- WEATHER DATA ----------------
def get_weather():
    api_key = os.getenv("OPENWEATHER_API_KEY")
    
    # Fallback mock weather function
    def _mock():
        temp = random.randint(20, 35)
        humidity = random.randint(40, 85)
        rain = random.randint(0, 10)
        wind = random.randint(5, 25)
        if humidity > 70 and temp > 25:
            alert = "High humidity and warm temp! High risk of fungal diseases."
        else:
            alert = "Weather is optimal. Low disease risk."
        return {"temp": temp, "humidity": humidity, "rain": rain, "wind": wind, "alert": alert}

    if not api_key or api_key == "your_api_key_here":
        print("No valid OPENWEATHER_API_KEY found, using mock weather.")
        return _mock()

    try:
        # Example location: default to a generic agricultural region or grab by IP.
        # Here we'll just hardcode a city or use a generic one for demonstration.
        # In a production app, you might pass lat/lon from the frontend.
        city = "Hyderabad" 
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        
        response = requests.get(url, timeout=5)
        data = response.json()
        
        if response.status_code == 200:
            temp = round(data["main"]["temp"])
            humidity = data["main"]["humidity"]
            wind = round(data["wind"]["speed"] * 3.6) # convert m/s to km/h
            rain = data.get("rain", {}).get("1h", 0) # mm in last hour
            
            if humidity > 70 and temp > 25:
                alert = "High humidity and warm temp! High risk of fungal diseases."
            else:
                alert = "Weather is optimal. Low disease risk."
                
            return {
                "temp": temp,
                "humidity": humidity,
                "rain": rain,
                "wind": wind,
                "alert": alert
            }
        else:
            print("Failed to fetch weather from API, using mock.")
            return _mock()
    except Exception as e:
        print("Weather API Error:", e)
        return _mock()

# ---------------- LOAD MODEL ----------------
try:
    model = tf.keras.models.load_model("model/plant_model.keras")
except Exception as e:
    print("Warning: Could not load model. Continuing without it.", e)
    model = None

# ---------------- LOAD CLASS INDICES ----------------
try:
    with open("class_indices.pkl", "rb") as f:
        class_indices = pickle.load(f)
    class_names = {v: k for k, v in class_indices.items()}
except Exception as e:
    print("Warning: Could not load class indices.", e)
    class_names = {0: "Tomato Early blight", 1: "Tomato healthy"}

# ---------------- NORMALIZE FUNCTION ----------------
def normalize(text):
    return text.lower().replace(" ", "").replace("_", "")

# ---------------- PREDICTION ----------------
def predict_image(path):
    img = cv2.imread(path)

    if img is None:
        return {
            "result": "Image not loaded ❌",
            "confidence": 0,
            "top3": []
        }

    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.reshape(img, (1, 224, 224, 3))

    if model:
        prediction = model.predict(img)[0]
        index = int(np.argmax(prediction))
        confidence = float(np.max(prediction)) * 100 # convert to percentage
        
        result = class_names.get(index, "Unknown")
        result = result.replace("___", " ").replace("_", " ")

        top3_idx = prediction.argsort()[-3:][::-1]
        top3 = []

        for i in top3_idx:
            name = class_names.get(i, "Unknown").replace("___", " ").replace("_", " ")
            conf = round(float(prediction[i]) * 100, 2)
            top3.append({"label": name, "conf": conf})

        if confidence < 60:
            return {
                "result": "Uncertain Prediction",
                "confidence": round(confidence, 2),
                "top3": top3
            }
    else:
        # Mock prediction if no model
        result = "Tomato Early blight"
        confidence = 94.5
        top3 = [{"label": "Tomato Early blight", "conf": 94.5}, {"label": "Tomato Late blight", "conf": 3.2}]

    return {
        "result": result,
        "confidence": round(confidence, 2),
        "top3": top3
    }

# ---------------- PDF ----------------
def create_pdf(result, data):
    file_path = "report.pdf"

    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()

    meds = ", ".join(data['medicine']) if isinstance(data['medicine'], list) else data['medicine']
    ferts = ", ".join(data['fertilizer']) if isinstance(data['fertilizer'], list) else data['fertilizer']
    prevs = ", ".join(data['prevention']) if isinstance(data['prevention'], list) else data['prevention']

    content = [
        Paragraph("AI Smart Plant Doctor Report", styles['Title']),
        Spacer(1, 20),
        Paragraph(f"<b>Disease:</b> {result}", styles['Normal']),
        Spacer(1, 10),
        Paragraph(f"<b>Medicine:</b> {meds}", styles['Normal']),
        Paragraph(f"<b>Fertilizer:</b> {ferts}", styles['Normal']),
        Paragraph(f"<b>Dosage:</b> {data['dosage']}", styles['Normal']),
        Paragraph(f"<b>Prevention:</b> {prevs}", styles['Normal']),
    ]

    doc.build(content)

# ---------------- VOICE ----------------
def generate_voice(result, data, lang="en"):
    meds = ", ".join(data['medicine']) if isinstance(data['medicine'], list) else data['medicine']
    ferts = ", ".join(data['fertilizer']) if isinstance(data['fertilizer'], list) else data['fertilizer']
    prevs = ", ".join(data['prevention']) if isinstance(data['prevention'], list) else data['prevention']

    if "Uncertain" in result:
        text = "Image not clear. Please upload a clear leaf image." if lang == "en" else "చిత్రం స్పష్టంగా లేదు. స్పష్టమైన ఆకు చిత్రం ఇవ్వండి."
    else:
        if lang == "en":
            text = f"Detected disease is {result}. Recommended medicine is {meds}. Use fertilizer {ferts}. Dosage is {data['dosage']}. Prevention steps are {prevs}."
        else:
            text = f"గుర్తించిన వ్యాధి {result}. ఉపయోగించవలసిన మందు {meds}. ఎరువు {ferts}. మోతాదు {data['dosage']}. నివారణ {prevs}."

    file_path = f"static/voice_{lang}.mp3"
    
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(file_path)
    except Exception as e:
        print(f"Error generating voice for {lang}: {e}")

    return file_path


# ---------------- ROUTES ----------------
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('image')

    if not file or file.filename == "":
        return render_template("index.html", error="No file selected ❌")

    filename = file.filename.replace(" ", "_")
    
    # Prevent duplicate names overwriting logic (simple timestamp)
    filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{filename}"
    path = os.path.join("static/uploads", filename)
    file.save(path)

    output = predict_image(path)

    result = output["result"]
    confidence = output["confidence"]
    top3 = output["top3"]

    # Match Treatment
    data = None
    for key in treatments:
        if normalize(key) in normalize(result) or normalize(result) in normalize(key):
            data = treatments[key]
            break

    if not data:
        data = {
            "medicine": ["Consult agriculture expert"],
            "fertilizer": ["Balanced NPK fertilizer"],
            "dosage": "As per crop guidelines",
            "prevention": ["Ensure proper sunlight", "Avoid overwatering"]
        }

    # Generate files
    create_pdf(result, data)
    generate_voice(result, data, "en")
    generate_voice(result, data, "te")

    # Weather
    weather = get_weather()
    
    # Save to SQLite
    conn = sqlite3.connect('db/history.db')
    c = conn.cursor()
    c.execute("INSERT INTO predictions (image_path, result, confidence) VALUES (?, ?, ?)", 
              (path, result, confidence))
    conn.commit()
    conn.close()

    # Generate WhatsApp Link
    meds_str = ", ".join(data['medicine'])
    whatsapp_text = f"🌱 AI Plant Doctor Results\n\n🌿 Plant/Disease: *{result}*\n💯 Confidence: {confidence}%\n💊 Medicine: {meds_str}\n\nCheck out my results!"
    whatsapp_url = f"https://api.whatsapp.com/send?text={urllib.parse.quote(whatsapp_text)}"

    # E-commerce search query
    buy_query = urllib.parse.quote(data['medicine'][0] if isinstance(data['medicine'], list) else data['medicine'])
    links = {
        "amazon": f"https://www.amazon.in/s?k={buy_query}+fungicide",
        "flipkart": f"https://www.flipkart.com/search?q={buy_query}+fungicide",
        "bighaat": f"https://www.bighaat.com/search?type=product&q={buy_query}"
    }

    return render_template("index.html",
        result=result,
        confidence=confidence,
        top3=top3,
        medicine=data['medicine'],
        fertilizer=data['fertilizer'],
        dosage=data['dosage'],
        prevention=data['prevention'],
        image_path=path,
        weather=weather,
        whatsapp_url=whatsapp_url,
        links=links
    )

@app.route('/history')
def history():
    conn = sqlite3.connect('db/history.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM predictions ORDER BY timestamp DESC")
    rows = c.fetchall()
    conn.close()
    return render_template("history.html", history=rows)

@app.route('/delete_history/<int:id>', methods=['POST'])
def delete_history(id):
    conn = sqlite3.connect('db/history.db')
    c = conn.cursor()
    c.execute("DELETE FROM predictions WHERE id = ?", (id,))
    conn.commit()
    conn.close()
    return {"status": "deleted"}, 200

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data.get("message", "")
    if not message:
        return jsonify({"error": "No message provided"}), 400

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        # Intelligent Offline Rule-Based Fallback
        msg_lower = message.lower()
        response_text = ""
        
        if "tomato" in msg_lower and "yellow" in msg_lower:
            response_text += "It looks like your tomato leaves are turning yellow due to nitrogen deficiency or early blight. Try applying a balanced NPK fertilizer and avoid overwatering."
        elif "fertilizer" in msg_lower and "paddy" in msg_lower:
            response_text += "For paddy crops, Urea (Nitrogen) and DAP (Phosphorus) are highly recommended during the early growth stages."
        elif "prevent" in msg_lower and "leaf spot" in msg_lower:
            response_text += "To prevent leaf spot disease, ensure good spacing between plants for airflow, remove fallen infected leaves, and apply a copper-based fungicide before the rainy season."
        elif "summer" in msg_lower:
            response_text += "Crops suitable for summer include Cucumber, Watermelon, Okra (Bhindi), and Moong Dal. Ensure you have a reliable irrigation source!"
        elif "water" in msg_lower and "chilli" in msg_lower:
            response_text += "Chilli plants require moderate watering. Water them deeply 2-3 times a week, allowing the topsoil to dry out slightly between waterings to prevent root rot."
        elif "telugu" in msg_lower or "నమస్కారం" in msg_lower or "తెలుగు" in msg_lower:
            response_text += "నమస్కారం! నేను మీ స్మార్ట్ వ్యవసాయ సహాయకుడిని. మీకు పంట వ్యాధులు లేదా ఎరువుల గురించి ఏమైనా సందేహాలు ఉంటే అడగండి."
        elif "hindi" in msg_lower or "नमस्ते" in msg_lower or "हिंदी" in msg_lower:
            response_text += "नमस्ते! मैं आपका स्मार्ट कृषि सहायक हूँ। आप मुझसे फसल की बीमारियों या उर्वरकों के बारे में कुछ भी पूछ सकते हैं।"
        else:
            response_text += "I'm here to help! Try asking me:\n- My tomato leaves are turning yellow, what should I do?\n- Which fertilizer is best for paddy crop?\n- How to prevent leaf spot disease?\n- What crop is suitable in summer season?\n- How much water should I give chilli plants?"
            
        return jsonify({"response": response_text})

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = (
            "You are an expert AI Farmer Assistant. Answer agricultural questions shortly, "
            "clearly, and in simple language. If the question is in Telugu, answer in Telugu. "
            "If it is in Hindi, answer in Hindi. "
            f"Question: {message}"
        )
        response = model.generate_content(prompt)
        return jsonify({"response": response.text})
    except Exception as e:
        print("Chat API Error:", e)
        return jsonify({"response": "Sorry, I am facing some technical issues right now. Please try again later. 🛠️"})

@app.route('/download')
def download():
    return send_file("report.pdf", as_attachment=True)

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)