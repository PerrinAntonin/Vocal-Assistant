import os
import json
import epitran
import unidecode
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import metrics, optimizers, losses
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Conv1D, Dense, Flatten,Lambda, Dropout, MaxPooling1D,LSTM,Input

import util.customCsv as uCsv
import util.editSongs as editSongs
import util.operationOnLists as operationOnLists


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def preProcessText(texts):
    textsplit=[]
    vocabTotal=[]
    for text in texts:
        text = unidecode.unidecode(text)
        text = text.lower()
        text = text.replace("\"", "")
        text = text.replace("(", "")
        text = text.replace(")", "")
        text = text.replace("_", "")
        text = text.replace("º", "")
        text = text.replace("=", "")
        text = text.replace("^", "")
        text = text.replace("{", "")
        text = text.replace("}", "")
        text = text.replace(";", ",")
        text = text.replace(":", " ")
        text = text.replace("|", "")
        text = text.replace(",", "")
        text = text.replace(".", "")
        text = text.strip()
        text = text2phonemes(text)
        text = " " + text + " "
        vocab = set(text)
        vocab = ''.join(vocab)
        
        vocabTotal.append(vocab)
        textsplit.append(text)
    flattened  = [val for sublist in vocabTotal for val in sublist]
    vocabTotal =list(set(flattened))
    print("vocabTotal(",len(vocabTotal),"): ",vocabTotal)
    return textsplit,vocabTotal

def text2phonemes(text):
    epi = epitran.Epitran('fra-Latn')
    # print(text +": "+str(epi.trans_list(text)))
    # return(epi.trans_list(text))
    return epi.transliterate(text)

# Cette fonction converti le texte en numerique 
def converttoInt(data,vocab):
    # integer encode input data
    integer_encoded = [vocab_to_int[char] for char in data]
    return integer_encoded

def get_batches(X, Y, batch_size):
    X = np.array(X)
    Y = np.array(Y)
    songs = []
    targets = []
    n_samples = X.shape[0]

    for idx in range(n_samples):
        song, target = balanceDataset(X[idx],Y[idx])
        songs.append(song)
        targets.append(target)
        if(len(songs)==batch_size):
            bufferSong = songs.copy()
            bufferTarget = targets.copy()
            songs = []
            targets = []
            yield bufferSong, bufferTarget

def balanceDataset(song,target):
    noUpdate = True
    index=0
    indexTarget=0
    targetbase = target.copy()
    ratio = len(song)/len(target)

    while len(target) < len(song):
        for i in range(int(ratio)):
            if len(target) < len(song):

                target.insert(index, targetbase[indexTarget])
                index+=1
                noUpdate = False
        indexTarget+=1
        if(noUpdate):
            if(len(target) != len(song)):
                print("pas bon: "+str(len(song)-len(target)))
            break
    if(len(target) != len(song)):
        print("pas bon: "+str(len(song)-len(target)))
    return song,target


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

@tf.function
def train_step(mfcc, target):
    # permet de surveiller les opérations réalisé afin de calculer le gradient    
    with tf.GradientTape() as tape:
        prediction = model(mfcc,training=True)
        # calcul de l'erreur en fonction de la prediction et des targets
        loss = tf.keras.losses.categorical_crossentropy(target,prediction)
    # calcul du gradient en fonction du loss
    # trainable_variables est la lst des variable entrainable dans le model
    gradients = tape.gradient(loss, model.trainable_variables)
    # changement des poids grace aux gradient
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    # ajout de notre loss a notre vecteur de stockage
    train_loss(loss)
    train_accuracy(target, prediction)

@tf.function
def valid_step(mfcc, target):
    prediction = model(mfcc)
    loss = tf.keras.losses.categorical_crossentropy(target, prediction)
    # Set the metrics for the test
    valid_loss(loss)
    valid_accuracy(target, prediction)


@tf.function
def predict(inputs):
    predictions = model.predict(inputs)
    return predictions



if __name__ == "__main__":
    print("TensorFlow version: {}".format(tf.__version__))
    print("Eager execution: {}".format(tf.executing_eagerly()))

    pathFile ="C:\\Users\\tompe\\Documents\\deeplearning\\Vocal-Assistant\\data\\clips\\"
    pathCsv = "C:\\Users\\tompe\\Documents\\deeplearning\\Vocal-Assistant\\data\\dev.tsv"
    pathValidaCsv = "C:\\Users\\tompe\\Documents\\deeplearning\\Vocal-Assistant\\data\\vali.tsv"
    exampleCsv = uCsv.customCsv(pathCsv)
    exampleCsv.readcsv()
    names,texts = exampleCsv.getContent()

    # A activer s'il y a des fichier mp3 dans la dataset
    #editSongs.mp3towav(names,pathFile)
    
    #Reduce for dev
    names = names[:10]

    texts = texts[:10]
    toolSong = editSongs.editSongs()
    
    mfccs = toolSong.loaMffcsFromWav(names,pathFile)
    print("\n All mfccs loaded")
    
    #toolSong.displayMffc(mfccs[2][2],texts[2])
    texts,vocab = preProcessText(texts)
    print("decoded_text example: ",texts[0])

    #Encodage du texte
    vocab_to_int = {l:i for i,l in enumerate(vocab)}
    int_to_vocab = {i:l for i,l in enumerate(vocab)}
    encoded_texts = []
    for text in texts:
        encoded_text =converttoInt(text,vocab)
        encoded_texts.append(encoded_text)
    print("encoded_text example: ",encoded_texts[0])
    print("decoded_text example: ",[int_to_vocab[char] for char in encoded_texts[0]])
    
    #decoded_text = [int_to_vocab[char] for char in encoded_texts[0]]

    
    #loss_object = tf.keras.losses.categorical_crossentropy()
    optimizer = tf.keras.optimizers.Adam(lr=0.001)
    # track the evolution
    # Loss
    train_loss = metrics.Mean(name='train_loss')
    valid_loss = metrics.Mean(name='valid_loss')
    # Accuracy
    train_accuracy = metrics.CategoricalAccuracy(name='train_accuracy')
    valid_accuracy = metrics.CategoricalAccuracy(name='valid_accuracy')

    model = createModel()
    model.summary()

    epochs = 15
    batch_size = 32
    actual_batch = 0
    model.reset_states()

    for epoch in range(epochs):
        print("\n epoch :", epoch)
        model.reset_states()
        actual_batch = 0
        for songs, targets in get_batches(mfccs, encoded_texts,batch_size):
            songs = np.array(songs)
            targets = np.array(targets)
            for song,target in zip(songs, targets):

                for x,y in zip(song, target):
                    x = np.expand_dims(x, axis=0)
                    y = tf.one_hot(np.array(y), len(vocab))
                    y = np.expand_dims(y, axis=0)
                    train_step(x, y)
                    #valid_step(x, y)
                
            template = '\r Batch {}/{}, Train Loss: {}, Train Accuracy: {}, Valid Loss: {}, Valid Accuracy: {}'
            print(template.format(actual_batch, len(names),
                            train_loss.result(), 
                            train_accuracy.result()*100,
                            valid_loss.result(), 
                            valid_accuracy.result()*100),
                            end="")
            actual_batch += batch_size
        # for songs, targets in get_batches(mfccs, encoded_texts,batch_size):
        #     valid_step(x, y)
        # template = '\r Batch {}/{}, Train Loss: {}, Train Accuracy: {}, Valid Loss: {}, Valid Accuracy: {}'
        # print(template.format(actual_batch, len(names),
        #                 train_loss.result(), 
        #                 train_accuracy.result()*100,
        #                 valid_loss.result(), 
        #                 valid_accuracy.result()*100),
        #                 end="")                
        #     model.reset_states()
        
    model.save("model_rnn.h5")

    with open("model_rnn_vocab_to_int", "w") as f:
        f.write(json.dumps(vocab_to_int))
    with open("model_rnn_int_to_vocab", "w") as f:
        f.write(json.dumps(int_to_vocab))