import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

import pandas as pd
import glob
from sklearn.cluster import KMeans

def find_wav_files(path):
  files=glob.glob(path+'*.wav')
  print("il y a ",len(files),"fichier wav")
  return np.array(files[:10])

def loadDataset():
    pathFile="C:/Users/anto/Documents/deepLearning/Vocal_Assistant/data/clips/"
    files=find_wav_files(pathFile)
    sounds = []
    for file in files:
        y, sr = librosa.load(file, sr=16000)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40) 
        sounds.append(mfccs)
    return sounds

def split_listtest(a_list):
    half = len(a_list)//2
    return a_list[:half]

def split_list(a_list,len_chunk):
    chunks = []
    for i in range(0, len(a_list), len_chunk):  
        chunks.append(a_list[i:i + len_chunk])
    chunks=chunks[:-1] 
    print("chunk",chunks[:-1])
    return chunks
        
    

#display test
y, sr = librosa.load("common_voice_fr_17300098.wav", sr=16000)
print("durée",librosa.core.get_duration(y=y, sr=sr))
print(len(y))

y1 = np.array(split_list(y,641))
print("shape of y", y1.shape)
mfccs = librosa.feature.mfcc(y=y1[1], sr=sr, n_mfcc=40)
mfccShape = np.array(mfccs).shape
print("shape of mfccs",mfccShape)
librosa.feature.mfcc(y=y, sr=sr)
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()

"""
audio = loadDataset()


km = KMeans(
    n_clusters=6, init='random',
    n_init=10, max_iter=300, 
    random_state=0
)

y_km = km.fit_predict(audio)

# plot the 3 clusters
plt.scatter(
    audio[y_km == 0, 0], audio[y_km == 0, 1],
    s=50, c='lightgreen',
    marker='s', edgecolor='black',
    label='cluster 1'
)

plt.scatter(
    audio[y_km == 1, 0], audio[y_km == 1, 1],
    s=50, c='orange',
    marker='o', edgecolor='black',
    label='cluster 2'
)

plt.scatter(
    audio[y_km == 2, 0], audio[y_km == 2, 1],
    s=50, c='lightblue',
    marker='v', edgecolor='black',
    label='cluster 3'
)

# plot the centroids
plt.scatter(
    km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
    s=250, marker='*',
    c='red', edgecolor='black',
    label='centroids'
)
plt.legend(scatterpoints=1)
plt.grid()
plt.show()
"""
"L’investissement dans l’industrie est en baisse."
#graph_spectrogram("common_voice_fr_17300098.wav")