import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.keras import metrics, optimizers, losses
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Conv1D, Dense, Flatten,Lambda, Dropout, MaxPooling1D,LSTM,Input

import util.customCsv as uCsv
import util.editSongs as editSongs
import util.operationOnLists as operationOnLists


def createModel():
    model = Sequential()
    
    model.add(Input(shape=(40,2), batch_size=1))
    model.add(Conv1D(8, 9, strides=4, padding="same", activation="elu"))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding="same"))

    model.add(LSTM(128, return_sequences=True, stateful=True))

    model.add(Flatten())
    model.add(tf.keras.layers.Dropout(.4))
    model.add(Dense(256, activation="elu"))
    model.add(tf.keras.layers.Dropout(.2))
    model.add(Dense(len(vocab), activation="softmax"))
    
    return model



def getsong():
    pathFile ="C:\\Users\\anto\\Documents\\deepLearning\\Vocal_Assistant\\data\\clips\\"
    pathFileV2 ="C:\\Users\\tompe\\Documents\\deepLearning\\Vocal_Assistant\\data\\clips\\"
    pathCsv = "C:/Users/anto/Documents/deepLearning/Vocal_Assistant/data/dev.tsv"
    pathCsvV2 = "C:/Users/tompe/Documents/deepLearning/Vocal_Assistant/data/dev.tsv"
    # a étudier plus tard
    #text2phonemes('hello')
    exampleCsv = uCsv.customCsv(pathCsvV2)
    exampleCsv.readcsv()
    names,texts = exampleCsv.getContent()
    #a activer s'il y a des fichier mp3 dans la dataset
    #mp3towav(names,pathFile)
    
    #Reduce for dev
    names= names[45]
    texts= texts[45]
    toolSong = editSongs.editSongs()
    print("name",names)
    mfccs= toolSong.loaMffcsFromWav([names],pathFileV2)
    
    print("mfccs load")
    return mfccs
def main():
    song  = getsong()
    print(np.array(song).shape)
    #model = createModel
    #model.load_weights("model_rnn.h5")
    model = tf.keras.models.load_model("model_rnn.h5")
    with open("model_rnn_vocab_to_int", "r") as f:
        vocab_to_int = json.loads(f.read())
    with open("model_rnn_int_to_vocab", "r") as f:
        int_to_vocab = json.loads(f.read())
        int_to_vocab = {int(key):int_to_vocab[key] for key in int_to_vocab}

        print(model(song[0]))



if __name__ == "__main__":
    main()