:construction: In Work :construction:

# Speech to text

A project of Speech Recognition in python using Tensorflow and keras.


## Model Architecture and Training method

I train the model using a network of recurrent neurons predicting a linear output. my dataset consists of audio associated with a sentence. I cut each sentence into phonemes to which I come to associate a sound.

## Model Architecture

The architecture is the following:
<ul>
    <li>One convolution of 8 filters (9*9) [Elu activation]</li>
    <li>Max pooling pool_size=[2,2]</li>
    <li>Lstm of 128 filters</li>
    <li>Flatten layer</li>
    <li>Dropout: 0.4</li>
    <li>Fully conected: 256 [Elu]</li>
    <li>Dropout: 0.2</li>
    <li>Fully conected: len(vocab) </li>
</ul>

