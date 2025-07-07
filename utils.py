import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

def preprocess_image(img, target_size=(224, 224)):
    img = cv2.resize(img, target_size)
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    lab = cv2.merge((cl, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return img

def predict_currency(image, mobilenet_model, efficientnet_model, svm_model, label_encoder):
    processed_img = preprocess_image(image)
    img_array = img_to_array(processed_img)

    mobilenet_feat = mobilenet_model.predict(np.expand_dims(mobilenet_preprocess(img_array.copy()), axis=0), verbose=0)
    efficientnet_feat = efficientnet_model.predict(np.expand_dims(efficientnet_preprocess(img_array.copy()), axis=0), verbose=0)

    hybrid_feature = np.concatenate((mobilenet_feat.flatten(), efficientnet_feat.flatten()))
    pred_index = svm_model.predict([hybrid_feature])[0]
    label = label_encoder.inverse_transform([pred_index])[0]
    return label
