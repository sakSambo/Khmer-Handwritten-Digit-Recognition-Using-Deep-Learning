from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)
CORS(app)  # Allow CORS for frontend

model = tf.keras.models.load_model("digit_model.keras")

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    image = Image.open(file).convert('L').resize((28, 28))  # Convert to grayscale
    image = np.array(image) / 255.0
    image = image.reshape(1, 28, 28, 1)

    prediction = model.predict(image)
    digit = int(np.argmax(prediction))

    return jsonify({'prediction': digit})

if __name__ == '__main__':
    app.run(debug=True)