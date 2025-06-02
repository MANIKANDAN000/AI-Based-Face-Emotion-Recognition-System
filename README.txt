# Face Emotion Recognition System

This project detects facial emotions using a CNN model trained on the FER-2013 dataset and integrates with a Flask web app for real-time webcam detection.

## Run Instructions
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model:
```bash
python app/train_model.py
```

3. Start the Flask server:
```bash
python app/server.py
```

## Features
- Train & evaluate CNN on FER-2013
- Real-time face detection
- Live emotion classification
- Web interface with styled CSS


1. (Optional but Recommended) Create a virtual environment:

conda create -n myapp python=3.11
conda activate myapp
2. Install dependencies
From your project folder (where requirements.txt is located):


pip install -r requirements.txt
If requirements.txt is incomplete or missing some modules, let me know and I can help generate a full one.

3. Run the Flask server
Assuming your entry point is server.py inside app/, run:


cd app
python server.py
You should then see something like:

csharp

 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
Open that URL in your browser to use the web app.

