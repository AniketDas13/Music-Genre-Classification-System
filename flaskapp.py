from flask import Flask, request
from flask_cors import CORS
import tensorflow as tf
import librosa
import numpy as np
import math, os

app = Flask(__name__)
CORS(app)

# Path to trained model
MODEL_PATH = r"D:\DWorkspace\Projects\MusicModel_Codes\MusicModel_v3_model_2.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Labels must match alphabetical order used in training
LABELS = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
]

# Parameters
SAMPLE_RATE = 22050
TRACK_DURATION = 30 # seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
NUM_MFCC = 20
N_FFT = 2048
HOP_LENGTH = 512
NUM_SEGMENTS = 10


def extract_segments(file_path):
    """Extract MFCC segments just like training pipeline"""
    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

    samples_per_segment = int(SAMPLES_PER_TRACK / NUM_SEGMENTS)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / HOP_LENGTH)

    mfcc_list = []
    for d in range(NUM_SEGMENTS):
        start = samples_per_segment * d
        finish = start + samples_per_segment

        mfcc = librosa.feature.mfcc(
            y=signal[start:finish],
            sr=sr,
            n_mfcc=NUM_MFCC,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        ).T

        if len(mfcc) == num_mfcc_vectors_per_segment:
            mfcc_list.append(mfcc)

    return np.array(mfcc_list, dtype=np.float32)


@app.route('/')
def hello_world():
    return {"result": "Hello, World!"}


@app.route('/prediction', methods=['POST'])
def genre_prediction():
    if 'the_file' not in request.files:
        return {"error": "No file uploaded"}, 400

    # Save uploaded file temporarily
    f = request.files['the_file']
    temp_path = os.path.join(os.getcwd(), "temp.wav")
    f.save(temp_path)

    try:
        segments = extract_segments(temp_path)
        if len(segments) == 0:
            return {"error": "File too short or no valid segments"}, 400

        # Predict with CNN
        probs = model.predict(segments, verbose=0)
        avg = probs.mean(axis=0)
        genre_idx = int(np.argmax(avg))
        genre = LABELS[genre_idx]

        return {"genre": genre}

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == '__main__':
    app.run(debug=True)