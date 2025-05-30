from flask import Flask
import tensorflow as tf
import librosa
import math
import os
import numpy as np
from flask import request
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

@app.route('/')
def hello_world():
    return {"result":'Hello, World!'}

@app.route('/prediction',methods=['POST'])
def genre_prediction():
    model = tf.keras.models.load_model('C:\\Users\\Aniket Das\\Desktop\\Workspace\\Seminar\\MusicModel_Codes\\MusicModel_v3_model.h5')
    labels={
        0:"Classical",
        1:"Blues",
        2:"Reggae",
        3:"Disco",
        4:"Rock",
        5:"Jazz",
        6:"Country",
        7:"Pop",
        8:"Hip Hop",
        9:"Metal"
    }

    mfcc_list=[]

    if request.method == 'POST':
        f = request.files['the_file']
        f.save('temp.wav')

    signal, sample_rate = librosa.load('temp.wav', sr=22050)

    ### Preprocess the uploaded file
    SAMPLE_RATE = 22050
    TRACK_DURATION = 30 # measured in seconds
    SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

    num_mfcc=20
    n_fft=2048
    hop_length=512
    num_segments=10

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)
    
    for d in range(num_segments):
        start = samples_per_segment * d
        finish = start + samples_per_segment

        # extract mfcc
        mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc = mfcc.T

        # store only mfcc feature with expected number of vectors
        if len(mfcc) == num_mfcc_vectors_per_segment:
            mfcc_list.append(mfcc.tolist())
    
    
    prediction_data = np.array(mfcc_list)

    prediction = model.predict_classes(prediction_data)

    prediction_except_duplicates = list(set(prediction))

    dict_preds={}

    for element in prediction_except_duplicates:
        count = 0
        for el in prediction:
            if el==element:
                count=count+1
        dict_preds[element] = count

    max_value = max(list(dict_preds.values()))
    genre_number_position = list(dict_preds.values()).index(max_value)
    genre_number = list(dict_preds.keys())[genre_number_position]

    os.remove("temp.wav")

    return {
        "genre":labels[genre_number]
    }
