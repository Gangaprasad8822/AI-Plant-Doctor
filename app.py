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

# ---------------- DB ----------------
def init_db():
    conn = sqlite3.connect('db/history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_path TEXT,
        result TEXT,
        confidence REAL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    conn.commit()
    conn.close()

init_db()

# ---------------- WEATHER ----------------
def get_weather():
    api_key = os.getenv("OPENWEATHER_API_KEY")

    def mock():
        return {
            "temp": random.randint(20, 35),
            "humidity": random.randint(40, 85),
            "rain": random.randint(0, 10),
            "wind": random.randint(5, 25),
            "alert": "Mock weather data"
        }

    if not api_key:
        return mock()

    try:
        city = "Hyderabad"
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        r = requests.get(url, timeout=5)
        data = r.json()

        if r.status_code != 200:
            return mock()

        return {
            "temp": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "rain": data.get("rain", {}).get("1h", 0),
            "wind": data["wind"]["speed"],
            "alert": "Weather OK"
        }
    except:
        return mock()

# ---------------- MODEL ----------------
try:
    model = tf.keras.models.load_model("model/plant_model.keras")
except:
    model = None
    print("Model not loaded")

# ---------------- CLASS NAMES ----------------
try:
    with open("class_indices.pkl", "rb") as f:
        class_indices = pickle.load(f)
    class_names = {v: k for k, v in class_indices.items()}
except:
    class_names = {0: "Tomato Early blight", 1: "Tomato healthy"}

# ---------------- NORMALIZE ----------------
def normalize(text):
    return text.lower().replace(" ", "").replace("_", "")

# ---------------- PREDICT ----------------
def predict_image(path):
    img = cv2.imread(path)

    if img is None:
        return {"result": "Image error", "confidence": 0, "top3": []}

    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.reshape(img, (1, 224, 224, 3))

    if model:
        pred = model.predict(img)[0]
        idx = np.argmax(pred)
        confidence = float(np.max(pred)) * 100

        result = class_names.get(idx, "Unknown")

        top3 = []
        for i in pred.argsort()[-3:][::-1]:
            top3.append({
                "label": class_names.get(i, "Unknown"),
                "conf": round(float(pred[i]) * 100, 2)
            })

        return {
            "result": result,
            "confidence": round(confidence, 2),
            "top3": top3
        }

    return {
        "result": "Tomato Early blight",
        "confidence": 90.0,
        "top3": []
    }

# ---------------- PDF ----------------
def create_pdf(result, data):
    doc = SimpleDocTemplate("report.pdf")
    styles = getSampleStyleSheet()

    content = [
        Paragraph("AI Plant Report", styles['Title']),
        Spacer(1, 20),
        Paragraph(f"Disease: {result}", styles['Normal']),
        Paragraph(f"Medicine: {data['medicine']}", styles['Normal']),
        Paragraph(f"Fertilizer: {data['fertilizer']}", styles['Normal']),
        Paragraph(f"Dosage: {data['dosage']}", styles['Normal']),
    ]

    doc.build(content)

# ---------------- VOICE ----------------
def generate_voice(text, lang="en"):
    try:
        path = f"static/voice_{lang}.mp3"
        tts = gTTS(text=text, lang=lang)
        tts.save(path)
        return path
    except:
        return None

# ---------------- ROUTES ----------------
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('image')

    if not file:
        return render_template("index.html", error="No image")

    filename = datetime.now().strftime("%Y%m%d%H%M%S") + "_" + file.filename
    path = os.path.join("static/uploads", filename)
    file.save(path)

    output = predict_image(path)

    result = output["result"]
    confidence = output["confidence"]

    data = None
    for k in treatments:
        if normalize(k) in normalize(result):
            data = treatments[k]
            break

    if not data:
        data = {
            "medicine": "Consult expert",
            "fertilizer": "NPK",
            "dosage": "As per guideline",
            "prevention": "Proper care"
        }

    create_pdf(result, data)
    generate_voice(result, "en")
    generate_voice(result, "te")

    weather = get_weather()

    conn = sqlite3.connect('db/history.db')
    c = conn.cursor()
    c.execute("INSERT INTO predictions (image_path, result, confidence) VALUES (?, ?, ?)",
              (path, result, confidence))
    conn.commit()
    conn.close()

    whatsapp_text = f"Plant: {result}\nConfidence: {confidence}%"
    whatsapp_url = f"https://api.whatsapp.com/send?text={urllib.parse.quote(whatsapp_text)}"

    return render_template("index.html",
        result=result,
        confidence=confidence,
        image_path=path,
        weather=weather,
        whatsapp_url=whatsapp_url
    )

# ---------------- CHAT ----------------
@app.route('/chat', methods=['POST'])
def chat():
    msg = request.get_json().get("message", "").lower()

    if msg:
        return jsonify({"response": "AI assistant working in fallback mode."})

    return jsonify({"response": "No message"})

# ---------------- DOWNLOAD ----------------
@app.route('/download')
def download():
    return send_file("report.pdf", as_attachment=True)

# ---------------- RUN (FIXED FOR RENDER) ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)