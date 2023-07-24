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
from tensorflow.keras import utils
from tensorflow.keras.utils import to_categorical
#import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as tl

import numpy as np
import math
import os

from tensorflow.keras.models import load_model


def generator(inputs,labels,
              n_of_suvr,
              activation='sigmoid'):
    """
    Generator Model

    Stack of MLP to generate suvr uptake data.
    Output activation is softmax.

    # Arguments
        inputs (Layer): Input layer of the generator (the z-vector)
        n_of_suvr (int): Target number of suvr uptake data samples to generate
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
    
    x = Dense(512)(x) # scale down 1/2
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.4)(x)
    x = Dense(2*n_of_suvr)(x)
    x = LeakyReLU(0.2)(x)
    x = Reshape((2,n_of_suvr))(x)

    if activation is not None:
        if activation == 'softmax':
            x = Softmax(axis=1)(x)
        else:
            x = Activation(activation)(x)

    x = Lambda(lambda x: x[:,1])(x)

    # generator output is the synthesized somatic mutation profile x
    return  Model([inputs, labels], x, name='generator')




def discriminator(inputs,labels,
                  n_of_suvr,
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
    
    y = Dense(4)(labels) # 8 is the latent size
    #y = Reshape((100, 1))(y)
    
    x = concatenate([x, y])
    
    x = Dense(512)(x)  # scale down
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.4)(x)
    x = Dense(256)(x)
    x = LeakyReLU(0.2)(x)
    x = Dense(128)(x)
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
    n_sample=100
    noise_input = np.random.uniform(-1.0, 1.0, size=[n_sample, 4]) # 4 is latent size
    
    if class_label is None:
        num_labels = 2
        noise_class = np.eye(num_labels)[np.random.choice(num_labels, n_sample)]
    else:
        noise_class = np.zeros((n_sample, 2))
        noise_class[:,class_label] = 1
    
    suvr_data = generator.predict([noise_input, noise_class])
    #genes = generator.predict(noise_input)
    noise_class2 = np.argmax(noise_class, axis=1)
    return suvr_data, noise_class2
