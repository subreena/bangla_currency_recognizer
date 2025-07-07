from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image
import joblib
import json
import os
import gdown
from tensorflow.keras.applications import MobileNetV3Large, EfficientNetB0
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.preprocessing.image import img_to_array

from utils import preprocess_image, predict_currency

app = Flask(__name__)

# === Step 1: Ensure model directory exists ===
model_dir = "model"
os.makedirs(model_dir, exist_ok=True)

# === Step 2: Download hybrid SVM model if missing ===
svm_model_path = os.path.join(model_dir, "hybrid_svm_model.joblib")
if not os.path.exists(svm_model_path):
    print("🔽 Downloading hybrid_svm_model.joblib from Google Drive...")
    file_id = "17Cw2Iek0llQRLEzAdimRCZeS7HE8jvsd"
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, svm_model_path, quiet=False)

# === Step 3: Load model files ===
try:
    svm_model = joblib.load(svm_model_path)
    label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.joblib"))

    with open(os.path.join(model_dir, "svm_class_indices.json"), "r") as f:
        class_indices = json.load(f)

except Exception as e:
    print("❌ Failed to load model files:", e)
    exit(1)

# === Step 4: Load CNN models ===
mobilenet_model = MobileNetV3Large(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
efficientnet_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))

# === Step 5: Routes ===
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['image']
        image_data = base64.b64decode(data.split(',')[1])
        image = Image.open(BytesIO(image_data)).convert('RGB')
        image_np = np.array(image)

        label = predict_currency(image_np, mobilenet_model, efficientnet_model, svm_model, label_encoder)
        return jsonify({'prediction': label})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# === Step 6: Run server ===
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
