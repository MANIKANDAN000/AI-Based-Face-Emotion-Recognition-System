🎭 AI-Based Face Emotion Recognition System
An intelligent facial emotion detection system that leverages Deep Learning and Computer Vision to identify human emotions from facial expressions in real-time. This project uses a Convolutional Neural Network (CNN) trained on the FER-2013 dataset and deployed via a Flask web interface.

🧠 Key Features
📷 Real-time face detection and emotion recognition using webcam or uploaded images.

🧠 Deep Learning with CNN architecture for high accuracy.

💾 Trained on the FER-2013 dataset (7 emotion classes: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral).

🌐 Web-based interface built with Flask and styled with attractive HTML/CSS.

📤 Option to upload images or use live camera feed.

⚙️ Tech Stack
Frontend: HTML, CSS, Bootstrap

Backend: Python, Flask

Libraries: OpenCV, TensorFlow/Keras, NumPy, Matplotlib

Dataset: FER-2013 (Facial Expression Recognition)

🧪 How It Works
Capture face from webcam or upload an image.

Preprocess the face region (grayscale, resize).

Pass the image through the trained CNN model.

Display the predicted emotion label with a confidence score.

🚀 Getting Started
bash
Copy
Edit
# Clone the repository
git clone https://github.com/your-username/ai-face-emotion-recognition.git
cd ai-face-emotion-recognition

# Install dependencies
pip install -r requirements.txt

# Run the Flask app
python app.py
Then open http://localhost:5000 in your browser.

📂 Project Structure
bash
Copy
Edit
/model              # Trained CNN model file (.h5)
/static             # CSS, images
/templates          # HTML templates (index.html)
/uploads            # Uploaded images (optional)
/app.py             # Flask app
/utils.py           # Helper functions
🎯 Emotion Classes Detected
Angry 😠

Disgust 🤢

Fear 😨

Happy 😀

Sad 😢

Surprise 😲

Neutral 😐

📷 Screenshots
(Optional section – add images showing the UI and prediction results)

🧑‍💻 Contributors
Manikandan R – Deep Learning & Flask Developer
