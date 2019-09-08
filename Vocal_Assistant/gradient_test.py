import numpy as np
import tensorflow as tf
from tensorflow.python.keras import metrics, optimizers, losses
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Conv1D, Dense, Flatten,Lambda, Dropout, MaxPooling1D,LSTM,Input

"""
x = tf.constant(3.0)
with tf.GradientTape() as g:
  g.watch(x)
  print("y:",x)
  y = x * x
  print("y:",y)
dy_dx = g.gradient(y, x) # Will compute to 6.0
print(g)


def createModel():
    model = Sequential()
    
    model.add(Input(shape=(2), batch_size=1))
    model.add(Conv1D(8, 9, strides=4, padding="same", activation="elu"))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding="same"))

    model.add(LSTM(128, return_sequences=True, stateful=True))
    #print(N)
    
    #model.add(Lambda(lambda x: x[:, -N:, :]))
    
    model.add(Flatten())
    model.add(Dropout(.6))
    model.add(Dense(256, activation="elu"))
    model.add(Dropout(.3))
    model.add(Dense(32, activation="softmax"))
    
    return model

model = createModel()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

inputs = [0,1]
targets = [0,1]
#targets = np.expand_dims(targets, axis=0)
#inputs = np.expand_dims(inputs, axis=0)

# permet de surveiller les opérations réalisé afin de calculer le gradient
with tf.GradientTape() as tape:
    # fait une prediction
    #predictions = model(inputs)
    print(" shape after creation model",targets)
    #print(" shape after creation model",predictions)
    # calcul de l'erreur en fonction de la prediction et des targets
    loss = loss_object(targets, inputs)
    print("calcul loss",loss)
# calcul du gradient en fonction du loss
# trainable_variables est la lst des variable entrainable dans le model
gradients = tape.gradient(loss, model.trainable_variables)
print("calcul gradient")
"""
lst = [1,2,3,2,1,3,8]
max_value = max(lst)
print("testtesttesttesttesttesttest",max_value)
max_index =lst.index(max_value)
print("max index", max_index)