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

from keras.utils.np_utils import to_categorical
#import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as tl

import numpy as np
import math
import os

def generator(inputs,labels,
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

    # Returns
        Model: Generator Model
    """
    # # default input is just 500-dim noise (z-code)
    # x = inputs

    # x = Dense(1024)(x)
    # x = LeakyReLU(0.2)(x)
    # x = Dropout(0.4)(x)
    # x = Dense(2*n_of_genes)(x)
    # x = LeakyReLU(0.2)(x)
    # x = Reshape((2,n_of_genes))(x)

    # if activation is not None:
    #     if activation == 'softmax':
    #         x = Softmax(axis=1)(x)
    #     else:
    #         x = Activation(activation)(x)

    # x = Lambda(lambda x: x[:,1])(x)

    # # generator output is the synthesized somatic mutation profile x
    # return Model(inputs, x, name='generator')
    
    ##################################### cGAN #################################
    
    x = concatenate([inputs, labels], axis=1)
    
    x = Dense(1024)(x)
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
    return  Model([inputs, labels], x, name='generator')




def discriminator(inputs,labels,
                  n_of_genes,
                  activation='sigmoid'):
    """
    Discriminator Model

    Stack of LeakyReLU-MLP to discriminate real from fake

    # Arguments
        inputs (Layer): Input layer of the discriminator (the image)
        activation (string): Name of output activation layer

    # Returns
        Model: Discriminator Model
    """
    # x = inputs
    # x = Dense(1024)(x)
    # x = LeakyReLU(0.2)(x)
    # x = Dropout(0.4)(x)
    # x = Dense(512)(x)
    # x = LeakyReLU(0.2)(x)
    # x = Dense(256)(x)
    # x = LeakyReLU(0.2)(x)

    # outputs = Dense(1)(x)
    
    # if activation is not None:
    #     print(activation)
    #     outputs = Activation(activation)(outputs)

    # return Model(inputs, outputs, name='discriminator')
    
    ##################################### cGAN #################################
    
    x = inputs
    
    y = Dense(100)(labels)
    #y = Reshape((100, 1))(y)
    
    x = concatenate([x, y])
    
    x = Dense(1024)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.4)(x)
    x = Dense(512)(x)
    x = LeakyReLU(0.2)(x)
    x = Dense(256)(x)
    x = LeakyReLU(0.2)(x)

    outputs = Dense(1)(x)
    
    if activation is not None:
        print(activation)
        outputs = Activation(activation)(outputs)
    
    
    #labels= Lambda(lambda labels: labels[:,:]) (labels)
    #outputs= Lambda(lambda outputs: outputs[:,:]) (outputs)
    
    #ll = tl.Layer ()(labels)
    #op = tl.Layer ()(outputs)
    
    #return  Model([inputs, K.constant(labels.astype('int'))], outputs, name='discriminator')    
    #return Model([inputs, ll], op, name='discriminator')
    return  Model([inputs, labels], outputs, name='discriminator')
    

def test_generator(generator, class_label=None):
    noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])
    
    if class_label is None:
        num_labels = 2
        noise_class = np.eye(num_labels)[np.random.choice(num_labels, 16)]
    else:
        noise_class = np.zeros((16, 2))
        noise_class[:,class_label] = 1
    
    genes = generator.predict([noise_input, noise_class])
    #genes = generator.predict(noise_input)
    return genes, noise_class
