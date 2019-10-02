import os
import util.customCsv as uCsv
import util.editSongs as editSongs
import util.operationOnLists as operationOnLists
import glob
import json
import time
import pydub
import epitran
import unidecode
import librosa
import librosa.display
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.keras import metrics, optimizers, losses
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Conv1D, Dense, Flatten,Lambda, Dropout, MaxPooling1D,LSTM,Input


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
    ipa=epi.transliterate(text)
    print(ipa)
    test = epitran.Epitran('fra-Latn',rev=True)
    print(test.reverse_transliterate(ipa))
    


# Elle a pour but de selectionner quel parti du song a gardé en fonction de prediction,
# afin de faire correspondre la longeur des targets au inputs.
# A FINIR!!!!!!!!
def select_pred(predictions, songs, targets):
    targets_len = len(targets)
    print("test",targets_len,"testest",len(predictions))
    sentence_predict= []
    for prediction in predictions:
        prediction = tf.keras.backend.get_value(prediction)
        prediction = list(prediction[0])
        max_value = max(prediction)
        #print("max value",max_value)
        max_index =prediction.index(max_value)
        #print("max index", max_index)
        #print("il a predit la lettre ",vocab[max_index])
        sentence_predict.append(vocab[max_index])
        correct_sentence_song = []
    i=0
    while i < (len(sentence_predict)-1): 
        if not (sentence_predict[i]==sentence_predict[i+1]):
            correct_sentence_song.append(songs[i])
        i += 1

    return correct_sentence_song

# Cette fonction converti le texte en numerique puis en encodage one hot
def converttoOneHot(data,vocab):
    # integer encode input data
    integer_encoded = [vocab_to_int[char] for char in data]
    # one hot encode
    """
    onehot_encoded = list()
    for value in integer_encoded:
        letter = [0 for _ in range(len(vocab))]
        letter[value] = 1
        onehot_encoded.append(letter)
        """
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
    model.add(Dropout(.6))
    model.add(Dense(256, activation="elu"))
    model.add(Dropout(.3))
    model.add(Dense(len(vocab), activation="softmax"))
    
    return model

@tf.function
def train_step(inputs, targets):
    # permet de surveiller les opérations réalisé afin de calculer le gradient    
    with tf.GradientTape() as tape:
        # fait une prediction
        predictions = model(inputs)
        print(" shape after prediction model of targets",targets.shape)
        print(" shape after prediction model of predictions",predictions.shape)
        # calcul de l'erreur en fonction de la prediction et des targets
        loss = loss_object(targets, predictions)
        print("calcul loss",loss)
    # calcul du gradient en fonction du loss
    # trainable_variables est la lst des variable entrainable dans le model
    gradients = tape.gradient(loss, model.trainable_variables)
    print("calcul gradient")
    # changement des poids grace aux gradient
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    print("etape optimizer")
    # ajout de notre loss a notre vecteur de stockage
    train_loss(loss)
    print("etape train loss")
    train_accuracy(targets, predictions)
    print("etape train accuracy")

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
    #if(len(sentence_predict)>len(targets)):
    sentence_predict,targets = operationOnLists.operationOnLists(sentence_predict,targets).divide_equitably()
    print("",len(sentence_predict),len(targets))
    
    #Seconde partie qui consiste a reduire le nombre de prediction equitablement
    
    return sentence_predict

@tf.function
def predict(inputs):
    # Make a prediction on all the batch
    predictions = model(inputs)
    return predictions



if __name__ == "__main__":
    pathFile ="C:\\Users\\anto\\Documents\\deepLearning\\Vocal_Assistant\\data\\clips\\"
    pathFileV2 ="C:\\Users\\tompe\\Documents\\dl6\\Vocal-Assistant\\data\\clips\\"
    pathCsv = "C:/Users/anto/Documents/deepLearning/Vocal_Assistant/data/dev.tsv"
    pathCsvV2 = "C:/Users/tompe/Documents/dl6/Vocal-Assistant/data/dev.tsv"
    # a étudier plus tard
    #text2phonemes('hello')
    exampleCsv = uCsv.customCsv(pathCsvV2)
    exampleCsv.readcsv()
    names,texts = exampleCsv.getContent()
    #a activer s'il y a des fichier mp3 dans la dataset
    #mp3towav(names,pathFile)

    #Reduce for dev
    names= names[:40]
    texts= texts[:40]
    toolSong = editSongs.editSongs()
    mfccs= toolSong.loaMffcsFromWav(names,pathFileV2)
    print("mfccs load")
    #displayMffc(mfccs[2][2],texts[2])
    texts,vocab = preProcessText(texts)

    #Encodage du texte
    vocab_to_int = {l:i for i,l in enumerate(vocab)}
    int_to_vocab = {i:l for i,l in enumerate(vocab)}
    encoded_texts = []
    for text in texts:
        encoded_text =converttoOneHot(text,vocab)
        encoded_texts.append(encoded_text)
    print("encoded_text",encoded_texts[0])
    print("decoded_text",int_to_vocab[np.argmax(encoded_texts[0])])
    
    #decoded_text = int_to_vocab[np.argmax(encoded_texts[0])]

    
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(lr=0.001)
    #track the evolution
    # Loss
    train_loss = metrics.Mean(name='train_loss')
    valid_loss = metrics.Mean(name='valid_loss')
    # Accuracy
    train_accuracy = metrics.SparseCategoricalCrossentropy(name='train_accuracy')
    valid_accuracy = metrics.SparseCategoricalCrossentropy(name='valid_accuracy')

    model = createModel()
    model.summary()

    epochs = 1
    batch_size = 256
    actual_batch = 0
    model.reset_states()
    print("generation des batchs.")
    batch_inputs, batch_targets = gen_batch(mfccs, encoded_texts,batch_size)

    for epoch in range(epochs):
        for song, target in zip(batch_inputs, batch_targets):
            song = np.array(song)
            target = np.array(target)
            print("shape of song",song.shape)
            print("shape of targets",target.shape)

            for x,y in zip(song, target):
                x = np.expand_dims(x, axis=0)
                train_step(x, y)
                template = '\r Batch {}/{}, Loss: {}, Accuracy: {}'
                print(template.format(actual_batch, len(texts),
                                train_loss.result(), 
                                train_accuracy.result()*100),
                                end="")
                actual_batch += batch_size
                
            model.reset_states()
    
    with open("model_rnn_vocab_to_int", "w") as f:
        f.write(json.dumps(vocab_to_int))
    with open("model_rnn_int_to_vocab", "w") as f:
        f.write(json.dumps(int_to_vocab))

"""
    for epoch in range(epochs):
        for batch_inputs, batch_targets in zip(mfccs, encoded_texts):
            for i in range(len(batch_inputs[1])):
                N=len(batch_targets)

                TrainX=np.array(batch_inputs[:, i])
                TrainX =np.float32(TrainX)
                TrainX= np.reshape(TrainX,(1,1, TrainX.shape[0]))
                batch_targets = np.array(batch_targets)
                batch_targets = np.expand_dims(batch_targets, axis=0)
                print(batch_targets.shape)
                print(batch_targets)
                #batch_targets= np.reshape(batch_targets,(1, batch_targets.shape))
                train_step(TrainX, batch_targets)
        template = '\r Epoch {}, Train Loss: {}, Train Accuracy: {}'
        print(template.format(epoch, train_loss.result(), train_accuracy.result()*100), end="")
        model.reset_states()
        stateful lstm
        """
