# -*- coding: utf-8 -*-
"""MusicModel_v3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1YeIXIAevE6XmNaeHxyN1UeOzQMIQc6GK
"""

import json
import os
import math
import librosa

from google.colab import drive
drive.mount('/content/drive')

DATASET_PATH = "/content/drive/MyDrive/Colab Notebooks/genres_original"
JSON_PATH = "data_10.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 30 # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

num_mfcc=20
n_fft=2048
hop_length=512
num_segments=10

data = {
        "mapping": [],
        "labels": [],
        "mfcc": []
    }

samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

# loop through all genre sub-folder
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(DATASET_PATH)):
        
        if dirpath is not DATASET_PATH:
            
            semantic_label = dirpath.split("\\")[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing: {}".format(semantic_label))
            
            for f in filenames:
		    
                file_path = os.path.join(dirpath, f)
                try:
                    signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
                except:
                    continue
               
                for d in range(num_segments):
                    
                    start = samples_per_segment * d
                    finish = start + samples_per_segment

                    # extract mfcc
                    mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                    mfcc = mfcc.T

                    # store only mfcc feature with expected number of vectors
                    if len(mfcc) == num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i-1)
                        print("{}, segment:{}".format(file_path, d+1))


    with open("data_final.json", "w") as fp:
        json.dump(data, fp, indent=4)

import json
with open("data_final.json", "r") as fp:
    data = json.load(fp)

import numpy as np 
x=np.array(data["mfcc"])
x1=np.array(data["mfcc"])

y=np.array(data["labels"])
y1=np.array(data["labels"])

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)
x_train_1,x_test_1,y_train_1,y_test_1 = train_test_split(x1,y1,test_size=0.3)

#Sequential

import tensorflow.keras as keras
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(x.shape[1],x.shape[2])),
    keras.layers.Dense(1024,activation='relu'),
    keras.layers.Dense(1024,activation='relu'),
    keras.layers.Dense(1024,activation='relu'),
    keras.layers.Dense(10,activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    x_train,y_train,batch_size=32,epochs=50
)

y_pred = model.predict(x_test)

model.evaluate(x_test,y_test,batch_size=32)

#CNN
import tensorflow.keras as keras
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D
from keras.models import Model

model2 = Sequential()

model2.add(Conv1D(120,5, padding = 'same',input_shape=(x1.shape[1],x1.shape[2]),activation = 'relu'))
model2.add(MaxPooling1D(pool_size=(6), padding = 'same'))
model2.add(Conv1D(120,5, padding = 'same' ,activation = 'relu'))
model2.add(Dropout(0.2)) 
model2.add(Flatten())
model2.add(Dense(10, activation = 'softmax'))

model2.compile(loss='sparse_categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model2.fit(x_train_1,y_train_1,batch_size = 32, epochs = 50, validation_data = (x_test_1, y_test_1))

model2.evaluate(x_test_1,y_test_1,batch_size=32)

model.save("/content/drive/MyDrive/Colab Notebooks/MusicModel_v3_model.h5")
model2.save("/content/drive/MyDrive/Colab Notebooks/MusicModel_v3_model_2.h5")

signal, sample_rate = librosa.load("/content/drive/MyDrive/Colab Notebooks/genres_original/disco/disco.00031.wav", sr=SAMPLE_RATE)

lst = []

for d in range(num_segments):
                    
                    start = samples_per_segment * d
                    finish = start + samples_per_segment

                    # extract mfcc
                    mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                    mfcc = mfcc.T

                    # store only mfcc feature with expected number of vectors
                    if len(mfcc) == num_mfcc_vectors_per_segment:
                        lst.append(mfcc.tolist())
                        print("{}, segment:{}".format("/content/drive/MyDrive/Colab Notebooks/genres_original/disco/disco.00031.wav", d+1))

x=np.array(lst)
x1 = np.array(lst)

var = keras.models.load_model("/content/drive/MyDrive/Colab Notebooks/MusicModel_v3_model.h5")
var1 = keras.models.load_model("/content/drive/MyDrive/Colab Notebooks/MusicModel_v3_model_2.h5")

final = np.argmax((var.predict(x) > 0.5).astype("int32"))
final_2 = np.argmax((var1.predict(x1) > 0.5).astype("int32"))
final_2 = final_2-(10*(final_2//10))

print("Using Linear model: Label of genre->", final)
print("Using CNN model: Label of genre->", final_2)

if final == 0:
  print(" The predicted genre is classical")
elif final == 1:
  print("The predicted genre is blues")
elif final == 2:
  print("The predicted genre is reggae")
elif final == 3:
  print("The predicted genre is disco")
elif final == 4:
  print("The predicted genre is rock")
elif final == 5:
  print("The predicted genre is jazz")
elif final == 6:
  print("The predicted genre is country")
elif final == 7:
  print("The predicted genre is pop")
elif final == 8:
  print("The predicted genre is hiphop")
else:
  print("The predicted genre is metal")

if final_2 == 0:
  print(" The predicted genre is classical")
elif final_2 == 1:
  print("The predicted genre is blues")
elif final_2 == 2:
  print("The predicted genre is reggae")
elif final_2 == 3:
  print("The predicted genre is disco")
elif final_2 == 4:
  print("The predicted genre is rock")
elif final_2 == 5:
  print("The predicted genre is jazz")
elif final_2 == 6:
  print("The predicted genre is country")
elif final_2 == 7:
  print("The predicted genre is pop")
elif final_2 == 8:
  print("The predicted genre is hiphop")
else:
  print("The predicted genre is metal")
  