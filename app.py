## VGG19 (transfer learning) for emotion detection

import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load the trained model
model = load_model('model_vgg19/emotion_model_vgg19.h5')

# Define emotion labels (ensure order matches model training)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(url_for('index'))

    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('index'))

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Load and preprocess the image
    img = image.load_img(file_path, target_size=(48, 48))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalization

    # Predict emotion
    predictions = model.predict(img_array)
    predicted_class = emotion_labels[np.argmax(predictions)]

    return render_template('index.html', prediction=predicted_class, image_path=file_path)

if __name__ == '__main__':
    app.run(debug=True)
