import os
import csv
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

def displayMffc(mfcc,text):
    print(text)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time')
    plt.colorbar()
    plt.title(text)
    plt.tight_layout()
    plt.show()

def find_mp3_files(names,path):
  files=glob.glob(path+'*.mp3')
  sounds=[]
  for file in files:
    buffer = file.replace(".mp3", ".wav")
    for name in names:
        pathcsv=path+name
        #print(pathcsv)
        #print(buffer)
        if(pathcsv==buffer):
            buffer = buffer.replace(".wav", ".mp3")
            sounds.append(buffer)
  print("il y a ",len(sounds),"fichier mp3")
  return np.array(sounds)

def mp3towav(names,path):
    files = find_mp3_files(names,path)
    for file in files:
        sound = pydub.AudioSegment.from_mp3(file)
        newFile=file.replace(".mp3", ".wav")
        #peut etre a mettre: ,bitrate='16k', parameters=["-acodec","pcm_u16le","-ac","1","-ar","8000"]
        sound.export(newFile, format="wav",bitrate='16k')
        os.remove(file)

def readcsv(pathCsv):
    with open(pathCsv , encoding="utf8") as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        names=[]
        texts=[]
        for line in tsvreader: 
            buffer = line[1].replace("mp3", "wav")
            names.append(buffer)
            texts.append(line[2])
    names=names[1:]
    texts=texts[1:]
    print("nb de csv",len(names))
    return names,texts

def loadWav(names,path,len_chunk = 641):
    wavFiles=[]
    for file in names:
        file=path+file
        y, sr = librosa.load(file, sr=16000)
        audios = split_list(y,len_chunk)
        song=[]
        for audio in audios:
            audio = np.array(audio)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
            #print(np.array(mfccs.shape))
            song.append(mfccs)

        wavFiles.append(song)
    return wavFiles

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
        text = text.replace("|", "")
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
    
def split_list(a_list,len_chunk):
    chunks = []
    for i in range(0, len(a_list), len_chunk):  
        chunks.append(a_list[i:i + len_chunk])
    chunks=chunks[:-1] 
    #print("chunk",chunks[:-1])
    return chunks

def select_pred(predictions, songs):
    sentence_predict= []
    for prediction in predictions:
        prediction = tf.keras.backend.get_value(prediction)
        prediction = list(prediction[0])
        max_value = max(prediction)
        print("max value",max_value)
        max_index =prediction.index(max_value)
        print("max index", max_index)
        print("il a predit la lettre ",vocab[max_index])
        sentence_predict.append(vocab[max_index])
        correct_sentence_song = []
        i=0
    while i < (len(sentence_predict)-1): 
        if not (sentence_predict[i]==sentence_predict[i+1]):
            correct_sentence_song.append(songs[i])
        i += 1
    return correct_sentence_song

def converttoOneHot(data,vocab):
    # integer encode input data
    integer_encoded = [vocab_to_int[char] for char in data]
    print(integer_encoded)
    # one hot encode
    onehot_encoded = list()
    for value in integer_encoded:
        letter = [0 for _ in range(len(vocab))]
        letter[value] = 1
        onehot_encoded.append(letter)
    return integer_encoded

def gen_batch(files,label_files, len_chunk = 641):      
    batch_x=[]
    batch_y=[]
    # Read in each input, perform preprocessing and get labels          
    for file, label_file in zip(files, label_files):
        #print("test")
        batch_x += [ file ]
        batch_y += [ label_file ] 
    #print(batch_x, batch_y)
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

def custom_train_step(inputs,targets):
    batch_input = np.float32(inputs)
    targets = np.array(targets)
    predictions = []
    for minibatch_input in batch_input:
        minibatch_input = np.expand_dims(minibatch_input, axis=0)
        prediction = predict(minibatch_input)
        predictions.append(prediction)
    sentence_predict = select_pred(predictions,batch_input)
    #print("il a predit cette phrase",sentence_predict)

    for letter,target in zip(sentence_predict,targets):
        #print("letter",letter)
        #print("target",target)
        letter = np.expand_dims(letter, axis=0)
        targets =[]
        """
        for test in target:
            test = np.expand_dims(test, axis=0)
            targets.append(test)
        targets  = np.array(targets)
        """
        train_step(letter,target)
    
    


@tf.function
def predict(inputs):
    # Make a prediction on all the batch
    predictions = model(inputs)
    return predictions



if __name__ == "__main__":
    pathFile ="C:\\Users\\anto\\Documents\\deepLearning\\Vocal_Assistant\\data\\clips\\"
    pathCsv = "C:/Users/anto/Documents/deepLearning/Vocal_Assistant/data/dev.tsv"
    #text2phonemes('hello')
    names,texts = readcsv(pathCsv)
    #mp3towav(names,pathFile)

    #reduce for dev
    names= names[:40]
    texts= texts[:40]

    mfccs=loadWav(names,pathFile)
    print("mfccs load")
    #displayMffc(mfccs[2][2],texts[2])
    texts,vocab = preProcessText(texts)

    vocab_to_int = {l:i for i,l in enumerate(vocab)}
    int_to_vocab = {i:l for i,l in enumerate(vocab)}
    encoded_texts = []
    for text in texts:
        encoded_text =converttoOneHot(text,vocab)
        encoded_texts.append(encoded_text)
    #print(encoded_texts[0])
    
    #decoded_text =[int_to_vocab[l] for l in encoded_text]
    #decoded_text = "".join(decoded_text)

    
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

    epochs = 20
    
    model.reset_states()
    batch_inputs, batch_targets = gen_batch(mfccs, encoded_texts)

    for epoch in range(epochs):
        for batch_input, batch_target in zip(batch_inputs, batch_targets):

            #batch_target = np.array(batch_target)
            #N = len(batch_inputs[0])
            #batch_input = np.expand_dims(batch_input, axis=0)
            #batch_targets = np.expand_dims(batch_targets, axis=0)
            custom_train_step(batch_input,batch_target)
                
            model.reset_states()
        template = '\r Epoch {}, Train Loss: {}, Train Accuracy: {}'
        print(template.format(epoch, train_loss.result(), train_accuracy.result()*100), end="")
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
