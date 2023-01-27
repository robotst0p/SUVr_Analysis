'''

Trains WGAN on ovarian somatic mutation profiles using Tensorflow Keras

This code follows "Advanced Deep Learning with Keras" training of WGAN at
https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

import numpy as np
import argparse

from lib import gan_cancer_ov as gan


def train(models, x_train, params):
    """
    Train function for the Discriminator and Adversarial Networks

    It first trains Discriminator with real and fake somatic mutations 
    (GAN-generated genes) for n_critic times.
    Discriminator weights are clipped as a requirement of Lipschitz constraint.
    Generator is trained next (via Adversarial) with fake somatic mutations
    pretending to be real.
    Generate sample genes per save_interval

    # Arguments
        models (list): Generator, Discriminator, Adversarial models
        x_train (tensor): Train mutation profiles
        params (list) : Networks parameters

    """
    
    generator, discriminator, adversarial = models
    # network parameters
    (batch_size, latent_size, n_critic, 
            clip_value, train_steps, model_name) = params
    # setting up a save interval to be every 500 steps
    save_interval = 2#500
    # number of elements in train dataset
    train_size = x_train.shape[0]
    # labels for real data
    real_labels = np.ones((batch_size, 1))
    for i in range(train_steps):
        # train discriminator n_critic times
        loss = 0
        acc = 0
        for _ in range(n_critic):
            # train the discriminator for 1 batch
            # 1 batch of real (label=1.0) and fake genes (label=-1.0)
            # randomly pick real genes from dataset
            rand_indexes = np.random.randint(0, train_size, size=batch_size)
            real_genes = x_train[rand_indexes]
            # generate fake genes from noise using generator
            # generate noise using uniform distribution
            noise = np.random.uniform(-1.0,
                                      1.0,
                                      size=[batch_size, latent_size])
            fake_genes = generator.predict(noise)

            # train the discriminator network
            # real data label=1, fake data label=-1
            real_loss, real_acc = discriminator.train_on_batch(real_genes,
                                                               real_labels)
            fake_loss, fake_acc = discriminator.train_on_batch(fake_genes,
                                                               -real_labels)
            # accumulate average loss and accuracy
            loss += 0.5 * (real_loss + fake_loss)
            acc += 0.5 * (real_acc + fake_acc)

            # clip discriminator weights to satisfy Lipschitz constraint
            for layer in discriminator.layers:
                weights = layer.get_weights()
                weights = [np.clip(weight,
                                   -clip_value,
                                   clip_value) for weight in weights]
                layer.set_weights(weights)

        # average loss and accuracy per n_critic training iterations
        loss /= n_critic
        acc /= n_critic
        log = "%d: [discriminator loss: %f, acc: %f]" % (i, loss, acc)
        # train the adversarial network for 1 batch
        # 1 batch of fake genes with label=1.0
        # only the generator is trained
        # generate noise using uniform distribution
        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, latent_size])
        # train the adversarial network
        # log the loss and accuracy
        loss, acc = adversarial.train_on_batch(noise, real_labels)
        log = "%s [adversarial loss: %f, acc: %f]" % (log, loss, acc)
        print(log)
        if (i + 1) % save_interval == 0:
            generator.save("weights/"+model_name+"_"+str(i)+".h5")

    # save the model after training the generator
    # the trained generator can be reloaded for future generation
    generator.save("weights/"+model_name + ".h5")


def wasserstein_loss(y_label, y_pred):
    """
    Implementation of a Wasserstein Loss with keras backend
    """
    return -K.mean(y_label * y_pred)


def build_and_train_models():
    # Load ovarian genes as a numpy array
    #x_train = np.load("ov_genes_converted.npy");
    x_train = np.random.randint(1, 10, (27, 100))

    model_name = "wgan_ov"
    # Setting up network parameters
    # The latent or z vector is 500-dim
    latent_size = 500
    # Hyper parameters
    n_critic = 5
    clip_value = 0.01
    batch_size = 12
    lr = 5e-5
    train_steps = 100#25000
    n_genes = x_train.shape[1]
    input_shape = (n_genes, )

    # Discriminator model
    inputs = Input(shape=input_shape, name='discriminator_input')
    # WGAN uses linear activation function 
    discriminator = gan.discriminator(inputs, n_genes, activation='linear')
    optimizer = RMSprop(lr=lr)
    discriminator.compile(loss=wasserstein_loss,
                          optimizer=optimizer,
                          metrics=['accuracy'])
    discriminator.summary()

    # Generator model
    input_shape = (latent_size, )
    inputs = Input(shape=input_shape, name='z_input')
    generator = gan.generator(inputs, n_genes, activation = 'softmax')
    generator.summary()

    # Adversarial model is a combination of generator and discriminator
    # Adversarial model should not update discriminator during adversarial training
    discriminator.trainable = False
    adversarial = Model(inputs,
                        discriminator(generator(inputs)),
                        name=model_name)
    adversarial.compile(loss=wasserstein_loss,
                        optimizer=optimizer,
                        metrics=['accuracy'])
    adversarial.summary()

    # train discriminator and adversarial networks
    models = (generator, discriminator, adversarial)
    params = (batch_size,
              latent_size,
              n_critic,
              clip_value,
              train_steps,
              model_name)
    train(models, x_train, params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load generator weights"
    parser.add_argument("-g", "--generator", help=help_)
    args = parser.parse_args()
    if args.generator:
        generator = load_model(args.generator)
        np.save("generated_ov"+str(time.strftime("%m|%d|%y_%H%M%S")) +".npy",\
                 gan.test_generator(generator));
    else:
        build_and_train_models()
