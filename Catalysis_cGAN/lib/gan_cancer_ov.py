'''

GAN model and util functions

This code follows "Advanced Deep Learning with Keras" util functions for WGAN at
https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras
'''


from tensorflow.keras.layers import Activation, Dense, Input, Lambda, Softmax
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate

import numpy as np
import math
import os

def generator(inputs,
              n_of_genes,
              activation='sigmoid'):
    """
    Generator Model

    Stack of MLP to generate counterpart genes.
    Output activation is softmax.

    # Arguments
        inputs (Layer): Input layer of the generator (the z-vector)
        n_of_genes (int): Target number of genes to generate
        activation (string): Name of output activation layer
        labels (tensor): Input labels
        codes (list): 2-dim disentangled codes for InfoGAN

    # Returns
        Model: Generator Model
    """
    # default input is just 500-dim noise (z-code)
    x = inputs

    x = Dense(512)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.4)(x)
    x = Dense(2*n_of_genes)(x)
    x = LeakyReLU(0.2)(x)
    x = Reshape((2,n_of_genes))(x)

    if activation is not None:
        if activation == 'softmax':
            x = Softmax(axis=1)(x)
        else:
            x = Activation(activation)(x)

    x = Lambda(lambda x: x[:,1])(x)

    # generator output is the synthesized somatic mutation profile x
    return Model(inputs, x, name='generator')


def discriminator(inputs,
                  n_of_genes,
                  activation='sigmoid'):
    """
    Discriminator Model

    Stack of LeakyReLU-MLP to discriminate real from fake

    # Arguments
        inputs (Layer): Input layer of the discriminator (the image)
        activation (string): Name of output activation layer
        num_labels (int): Dimension of one-hot labels for ACGAN & InfoGAN
        num_codes (int): num_codes-dim Q network as output 
                    if StackedGAN or 2 Q networks if InfoGAN
                    

    # Returns
        Model: Discriminator Model
    """
    x = inputs
    x = Dense(1024)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.4)(x)
    x = Dense(256)(x)
    x = LeakyReLU(0.2)(x)

    outputs = Dense(1)(x)
    if activation is not None:
        print(activation)
        outputs = Activation(activation)(outputs)

    return Model(inputs, outputs, name='discriminator')

def test_generator(generator):
    noise_input = np.random.uniform(-1.0, 1.0, size=[1000, 500])
    genes = generator.predict(noise_input)
    return genes


