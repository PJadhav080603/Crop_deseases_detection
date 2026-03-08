from flask import Flask, request, render_template, jsonify, url_for
from tensorflow.keras.models import load_model  # For TensorFlow
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Load the model
model = load_model(r"C:\Users\prati\Desktop\mmini\New folder (2)\New folder (2)\plant_disease_model.h5")

# Preprocessing function
def preprocess_image(image_path):
    # Assuming the image needs to be resized and normalized for the model
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Change according to your model's input size
    img = np.array(img) / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    # Save the uploaded image
    img_path = os.path.join('static/uploaded_images', file.filename)
    os.makedirs(os.path.dirname(img_path), exist_ok=True)  # Create the directory if it doesn't exist
    file.save(img_path)
    
    # Preprocess the image
    processed_img = preprocess_image(img_path)
    
    # Make the prediction
    prediction = model.predict(processed_img)
    
    # Assuming your model outputs class probabilities
    predicted_class = np.argmax(prediction, axis=1)
    
    # Map this to the actual disease names
    class_labels = ['Pepper Bell having Bacterial Spot', 'Pepper Bell is Healthy', 'Potato with Early blight ', 'Potato is Healthy ','Potato with Late blight']  # Update with actual class names
    result = class_labels[predicted_class[0]]
    
    # Return result as JSON
    return jsonify({'prediction': result, 'image_url': url_for('static', filename='uploaded_images/' + file.filename)})

if __name__ == '__main__':
    app.run(debug=True)
