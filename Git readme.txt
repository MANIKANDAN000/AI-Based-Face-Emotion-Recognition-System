# Emotion Recognition using AI

This project implements a deep learning-based emotion recognition system using a Convolutional Neural Network (CNN) trained on the FER-2013 dataset. The system predicts emotions from facial expressions in images and provides predictions for emotions like "Angry", "Happy", "Sad", etc.

The web app uses Flask as the backend to serve the trained emotion recognition model, allowing users to upload face images and predict emotions. The frontend is designed with HTML, CSS, and JavaScript for a clean and modern user interface.

## Features

- **Emotion Prediction**: Upload an image and the system will predict the emotion expressed on the face.
- **User-Friendly Interface**: Clean, responsive design with simple interaction.
- **Real-time Prediction**: Instant emotion classification on uploaded images using a trained deep learning model.
- **Multiple Emotions Supported**: Supports 7 different emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.

## Project Structure

your_project/
├── app.py # Flask backend for handling image upload and emotion prediction
├── dataset/
│ └── archive/
│ ├── train/ # Training dataset images
│ └── test/ # Test dataset images
├── model/
│ └── emotion_model.h5 # Trained emotion recognition model
├── static/
│ └── style.css # Custom CSS for frontend styling
├── templates/
│ └── index.html # Frontend HTML for displaying UI
├── uploads/ # Folder to temporarily store uploaded images

markdown
Copy
Edit

## Prerequisites

Before running the application, ensure you have the following software installed:

- Python 3.x
- Flask
- TensorFlow
- Keras
- NumPy
- Matplotlib
- PIL (Pillow)
- OpenCV

You can install the necessary Python libraries using the following commands:

```bash
pip install flask tensorflow keras numpy matplotlib pillow opencv-python
Setup Instructions
Clone the Repository

Clone this repository to your local machine:

bash
Copy
Edit
git clone https://github.com/yourusername/emotion-recognition.git
cd emotion-recognition
Download or Train the Model

If you already have the emotion_model.h5 file, place it in the model/ folder.

If not, you can train the model using the provided train_model.py script.

To train the model:

bash
Copy
Edit
python train_model.py
This will train the model on the FER-2013 dataset and save it as emotion_model.h5 in the model/ folder.

Run the Flask App

Once the model is trained, run the Flask application:

bash
Copy
Edit
python app.py
The app will be accessible at: http://127.0.0.1:5000/

Upload an Image for Prediction

Open the app in your browser.

Upload a face image to predict the emotion on the face.

The predicted emotion will be displayed along with the uploaded image.

How It Works
Backend (Flask)
The Flask backend handles image uploads and invokes the trained CNN model to predict emotions.

The model is loaded from model/emotion_model.h5 using TensorFlow and Keras.

The uploaded image is preprocessed (resized and normalized) before being fed to the model.

The model outputs a prediction of the emotion, which is then displayed on the web page.

Frontend (HTML/CSS)
The frontend is a simple HTML form that allows users to upload an image.

The CSS provides an attractive and modern design with smooth animations and responsive layout.

Emotions Supported
The model classifies the following 7 emotions:

Angry

Disgust

Fear

Happy

Neutral

Sad

Surprise

Example Prediction


Predicted Emotion: Happy
Customization
Dataset: The current dataset used is the FER-2013 dataset, which is commonly used for facial emotion recognition tasks. You can use your own dataset by placing the images in the train/ and test/ directories.

Model Tuning: You can tune the CNN model architecture (layers, activation functions, etc.) in the train_model.py script to improve accuracy or reduce overfitting.

Contributing
Feel free to fork this repository and submit pull requests. Contributions to improve the model, frontend, or backend are always welcome!

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgements
The FER-2013 dataset is used for training the emotion recognition model.

This project uses TensorFlow and Keras for deep learning, Flask for the web application, and OpenCV for image processing.

