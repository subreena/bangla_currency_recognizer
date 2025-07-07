from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image
import joblib
import json
import os
import gdown  #  for downloading from Google Drive
from tensorflow.keras.applications import MobileNetV3Large, EfficientNetB0
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.preprocessing.image import img_to_array

from utils import preprocess_image, predict_currency

app = Flask(__name__)

# Step: Download model if not already present
model_path = "model/hybrid_svm_model.joblib"
model_dir = "model"
os.makedirs(model_dir, exist_ok=True)
# https://drive.google.com/file/d/17Cw2Iek0llQRLEzAdimRCZeS7HE8jvsd/view?usp=sharing
if not os.path.exists(model_path):
    print("🔽 Downloading model from Google Drive...")
    file_id = "17Cw2Iek0llQRLEzAdimRCZeS7HE8jvsd"  # <-- ✅ Replace with your file ID
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, model_path, quiet=False)

# Load models and encoders
svm_model = joblib.load("model/hybrid_svm_model.joblib")
label_encoder = joblib.load("model/label_encoder.joblib")
with open("model/svm_class_indices.json", "r") as f:
    class_indices = json.load(f)

mobilenet_model = MobileNetV3Large(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
efficientnet_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['image']
    image_data = base64.b64decode(data.split(',')[1])
    image = Image.open(BytesIO(image_data)).convert('RGB')
    image_np = np.array(image)

    label = predict_currency(image_np, mobilenet_model, efficientnet_model, svm_model, label_encoder)
    return jsonify({'prediction': label})
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    # Run on all interfaces, so Render can reach it
    app.run(host='0.0.0.0', port=port, debug=True)