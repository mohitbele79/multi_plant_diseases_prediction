from flask import Flask, render_template, request, redirect, url_for
import json
import tensorflow as tf
import numpy as np
import base64
import os
from PIL import Image
from io import BytesIO

app = Flask(__name__)

# Load the Model
MODEL_PATH = "cnn_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Ensure the uploads folder exists
UPLOAD_FOLDER = "static/uploads/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def preprocess_image(image):
    """Preprocess the image for model prediction"""
    img = image.convert('RGB')  # Convert to RGB format
    img = img.resize((256, 256))  # Resize to model input size
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route('/')
def home():
    with open("static/diseases/diseases.json", "r") as file:
        diseases = json.load(file)
    return render_template('home.html', diseases=diseases)

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        if 'image' in request.form:
            image_data = request.form['image'].split(',')[1]  # Remove base64 header
            image_bytes = base64.b64decode(image_data)  # Decode base64
            
            # Convert image bytes to PIL Image
            image = Image.open(BytesIO(image_bytes))
            
            # Generate a unique filename
            filename = f"user_uploaded_{np.random.randint(1000, 9999)}.jpg"
            image_path = os.path.join(UPLOAD_FOLDER, filename)
            
            # Save the uploaded image
            image.save(image_path)

            # Process image for model prediction
            img_array = preprocess_image(image)
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction, axis=1)[0]

            # Disease mapping
            disease_mapping = {
                0: "Apple Apple scab", 1: "Apple Black rot", 2: "Apple Cedar apple rust", 
                3: "Apple healthy", 4: "Blueberry healthy", 5: "Cherry healthy", 
                6: "Cherry Powdery mildew", 7: "Corn Cercospora leaf spot", 
                8: "Corn Common rust", 9: "Corn healthy", 10: "Corn Northern Leaf Blight",
                11: "Grape Black rot", 12: "Grape Esca (Black Measles)",
                13: "Grape Leaf blight (Isariopsis Leaf Spot)", 14: "Grape healthy",
                15: "Orange Haunglongbing (Citrus greening)", 16: "Peach Bacterial spot",
                17: "Peach healthy", 18: "Pepper bell Bacterial spot",
                19: "Pepper bell healthy", 20: "Potato Early blight",
                21: "Potato Late blight", 22: "Potato healthy", 23: "Raspberry healthy",
                24: "Soybean healthy", 25: "Squash Powdery mildew",
                26: "Strawberry Leaf scorch", 27: "Strawberry healthy",
                28: "Tomato Bacterial spot", 29: "Tomato Early blight",
                30: "Tomato Late blight", 31: "Tomato Leaf Mold",
                32: "Tomato Septoria leaf spot", 33: "Tomato Spider mites (Two-Spotted Spider Mite)",
                34: "Tomato Target Spot", 35: "Tomato Tomato Yellow Leaf Curl Virus",
                36: "Tomato Tomato mosaic virus", 37: "Tomato healthy"
            }
            disease = disease_mapping.get(predicted_class, "Unknown Disease")

            # Fetch disease info and solutions
            with open("static/diseases/diseases_info.json", "r") as file:
                prediction_data = json.load(file)
            disease_data = next((item for item in prediction_data if item["name"] == disease), {})
            disease_information = disease_data.get("info", "No information available.")
            solutions = disease_data.get("solutions", ["No solutions available."])
            confidence = round(np.max(prediction) * 100, 2)

            # Send data to result page
            return render_template(
                'result.html',
                image_url=url_for('static', filename=f"uploads/{filename}"),  # Pass correct image URL
                disease=disease,
                confidence=f"{confidence}%",
                disease_information=disease_information,
                solutions=solutions
            )
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
