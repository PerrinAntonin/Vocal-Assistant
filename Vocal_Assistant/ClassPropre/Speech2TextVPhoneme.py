import os
import glob
import json
import time
import pydub
import librosa
import epitran
import unidecode
import numpy as np
import librosa.display
import tensorflow as tf
import matplotlib.pyplot as plt
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
        text = text.replace("ç", "c")
        text = text.replace("à", "a")
        text = text.replace("º", "")
        text = text.replace("=", "")
        text = text.replace("^", "")
        text = text.replace("{", "")
        text = text.replace("}", "")
        text = text.replace(";", ",")
        text = text.replace(":", " ")
        text = text.replace("'", " ")
        text = text.replace("|", "")
        text = text.replace(",", "")
        text = text.strip()
        text = text2phonemes(text)
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
    #return(epi.trans_list(text))
    return epi.transliterate(text)

# Elle a pour but de selectionner quel parti du song a gardé en fonction de prediction,
# afin de faire correspondre la longeur des targets au inputs.
def select_pred(predictions, songs, targets):
    targets_len = len(targets)
    #print("test",targets_len,"testest",len(predictions))
    sentence_predict= []
    for prediction in predictions:
        prediction = tf.keras.backend.get_value(prediction)
        
        prediction = list(prediction[0])
        max_index = np.argmax(prediction)
        #print("max index", max_index)
        #print("il a predit la lettre ",vocab[max_index])
        sentence_predict.append(vocab[max_index])
        correct_sentence_song = []
    i=0
    while (i < (len(sentence_predict)-1)):
        if (len(targets)>len(correct_sentence_song)):
            if not (sentence_predict[i]==sentence_predict[i+1]):
                correct_sentence_song.append(songs[i])
        i += 1

    return correct_sentence_song

# Cette fonction converti le texte en numerique 
def converttoInt(data,vocab):
    # integer encode input data
    integer_encoded = [vocab_to_int[char] for char in data]
    return integer_encoded

def gen_batch(files,label_files,batch_size=256):      
    batch_x=[]
    batch_y=[]

    print("Selection des bons echantillions de song a utiliser.")
    for file, label_song in zip(files, label_files):
        correct_sample_song = preProcessAudio(file, label_song)
        
        batch_x += [ correct_sample_song ]
        batch_y += [ label_song ] 
    return batch_x, batch_y

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
def train_step(inputs, targets):
    # permet de surveiller les opérations réalisé afin de calculer le gradient    
    with tf.GradientTape() as tape:
        # fait une prediction
        predictions = model(inputs)
        #print(" shape after prediction model of targets",targets.shape)
        #print(" shape after prediction model of predictions",predictions.shape)
        # calcul de l'erreur en fonction de la prediction et des targets
        loss = tf.keras.losses.categorical_crossentropy(targets, predictions)
        #print("calcul loss",loss)
    # calcul du gradient en fonction du loss
    # trainable_variables est la lst des variable entrainable dans le model
    gradients = tape.gradient(loss, model.trainable_variables)
    #print("calcul gradient")
    # changement des poids grace aux gradient
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    #print("etape optimizer")
    # ajout de notre loss a notre vecteur de stockage
    train_loss(loss)
    #print("etape train loss")
    train_accuracy(targets, predictions)
    #print("etape train accuracy")

def preProcessAudio(inputs,targets):
    #premiere partie qui consiste a enlever les doublons
    batch_input = np.float32(inputs)
    targets = np.array(targets)
    predictions = []

    for input in batch_input:
        input = np.expand_dims(input, axis=0)
        prediction = predict(input)
        predictions.append(prediction)
    sentence_predict = select_pred(predictions, batch_input, targets)
    # A CHANGER!!!!
    if(len(sentence_predict)>len(targets)):
        sentence_predict,targets = operationOnLists.operationOnLists(sentence_predict,targets).divide_equitably()
    if(len(sentence_predict)<len(targets)):
         print("y'a un probbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
    print("Finale",len(sentence_predict),len(targets))    


    #Seconde partie qui consiste a reduire le nombre de prediction equitablement
    
    return sentence_predict

@tf.function
def predict(inputs):
    # Make a prediction on all the batch
    predictions = model(inputs)
    return predictions



if __name__ == "__main__":

    pathFile ="C:\\Users\\anto\\Documents\\deepLearning\\Vocal_Assistant\\data\\clips\\"
    pathFileV2 ="C:\\Users\\tompe\\Documents\\deepLearning\\Vocal_Assistant\\data\\clips\\"
    pathCsv = "C:/Users/anto/Documents/deepLearning/Vocal_Assistant/data/dev.tsv"
    pathCsvV2 = "C:/Users/tompe/Documents/deepLearning/Vocal_Assistant/data/dev.tsv"

    exampleCsv = uCsv.customCsv(pathCsvV2)
    exampleCsv.readcsv()
    names,texts = exampleCsv.getContent()

    #a activer s'il y a des fichier mp3 dans la dataset
    #mp3towav(names,pathFile)
    
    #Reduce for dev
    names= names[:500]
    texts= texts[:500]
    toolSong = editSongs.editSongs()
    
    mfccs= toolSong.loaMffcsFromWav(names,pathFileV2)
    print("mfccs load")
    
    #displayMffc(mfccs[2][2],texts[2])
    texts,vocab = preProcessText(texts)
    print("testtetsttetsttt",texts[0])

    #Encodage du texte
    vocab_to_int = {l:i for i,l in enumerate(vocab)}
    int_to_vocab = {i:l for i,l in enumerate(vocab)}
    encoded_texts = []
    for text in texts:
        encoded_text =converttoInt(text,vocab)
        encoded_texts.append(encoded_text)
    print("encoded_text",encoded_texts[0])
    print("decoded_text",int_to_vocab[np.argmax(encoded_texts[0])])
    
    #decoded_text = int_to_vocab[np.argmax(encoded_texts[0])]

    
    #loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    #loss_object = tf.keras.losses.categorical_crossentropy()
    optimizer = tf.keras.optimizers.Adam(lr=0.001)
    #track the evolution
    # Loss
    train_loss = metrics.Mean(name='train_loss')
    valid_loss = metrics.Mean(name='valid_loss')
    # Accuracy
    train_accuracy = metrics.CategoricalAccuracy(name='train_accuracy')
    valid_accuracy = metrics.CategoricalAccuracy(name='valid_accuracy')

    model = createModel()
    model.summary()

    epochs = 15
    batch_size = 256
    actual_batch = 0
    model.reset_states()
    print("generation des batchs.")
   
    songs, targets =gen_batch(mfccs, encoded_texts,batch_size)
    print("batchSize : ",batch_size)
    for epoch in range(epochs):
        print("\n epoch :", epoch)
        for song, target in zip(songs, targets):
            song = np.array(song)
            target = np.array(target)
            #print("shape of song",song.shape)
            #print("shape of targets",target.shape)

            for x,y in zip(song, target):
                x = np.expand_dims(x, axis=0)
                y = tf.one_hot(y, len(vocab))
                train_step(x, y)
                template = '\r Batch {}/{}, Loss: {}, Accuracy: {}'
                print(template.format(actual_batch, len(texts),
                                train_loss.result(), 
                                train_accuracy.result()*100),
                                end="")
                actual_batch += batch_size
                
            model.reset_states()
    
    model.save("model_rnn.h5")

    with open("model_rnn_vocab_to_int", "w") as f:
        f.write(json.dumps(vocab_to_int))
    with open("model_rnn_int_to_vocab", "w") as f:
        f.write(json.dumps(int_to_vocab))
