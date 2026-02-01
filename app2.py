import io
import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)

# Load EfficientNetB0 model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_model_finetuned.keras')

def prepare_image(image_bytes):
    # 1. Load image and convert to RGB
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # 2. Resize to EfficientNetB0's native 224x224
    img = img.resize((224, 224))
    
    # 3. Convert to array and add batch dimension
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # IMPORTANT: EfficientNetB0 has internal rescaling (0-255).
    # Do NOT divide by 255.0 here unless you removed that layer during training.
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    # Read the file from the request
    file = request.files['image'].read()
    
    # Preprocess
    processed_image = prepare_image(file)
    
    # Run Inference
    prediction = model.predict(processed_image)[0][0]
    
    # Determine Label (Assuming 0=Real, 1=Fake)
    label = "Fake" if prediction > 0.4 else "Real"
    confidence = float(prediction) if label == "Fake" else float(1 - prediction)
    
    return jsonify({
        'status': 'success',
        'prediction': label,
        'confidence': f"{confidence:.2%}"
    })

# 3. Standard Flask entry point 
 if __name__ == '__main__':
     app.run(host='0.0.0.0', port=9696, debug=False)
