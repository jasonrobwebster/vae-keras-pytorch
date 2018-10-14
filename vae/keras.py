# keras implementation of a variational autoencoder

import numpy as np

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, Dense, Flatten, Reshape
from keras.datasets import mnist


def vae_model(n=2, hidden_dim=512):
    """Builds a variational autoencoder

    Params
    ------

    n : int
        The dimensionality of the autoencoder

    hidden_dim: int
        Dimensionality of the decoder hidden layers

    Returns
    -------

    model: keras model
        A keras model of a encoder-decoder model

    enc_model: Keras model
        A keras model of the vae encoder

    dec_model: Keras model
        A keras model of the vae decoder
    """
    # encoder model
    enc_in = Input(shape=(28, 28, 1), name='enc_in')
    #y = Conv2D(64, 7, padding='same', name='enc_conv1', activation='relu')(enc_in)
    #y = Conv2D(64, 7, padding='same', name='enc_conv2', activation='relu')(y)
    y = Flatten(name='enc_flatten')(enc_in)
    y = Dense(2*hidden_dim, name='enc_hidden1', activation='relu')(y)
    y = Dense(hidden_dim, name='enc_hidden2', activation='relu')(y)
    z_mean = Dense(n, name='enc_mean')(y)
    z_var = Dense(n, name='enc_var')(y)

    enc_model = Model(enc_in, [z_mean, z_var], name='Encoder')

    # decoder model
    dec_in = Input(shape=(n,), name='dec_input')
    y = Dense(hidden_dim, name='dec_hidden1', activation='relu')(dec_in)
    y = Dense(2*hidden_dim, name='dec_hidden2', activation='relu')(y)
    y = Dense(28*28*1, name='dec_output', activation='sigmoid')(y)
    dec_out = Reshape((28, 28, 1), name='dec_image')(y)

    dec_model = Model(dec_in, dec_out, name='Decoder')

    # loss function
    def vae_loss(y_true, y_pred):
        im_loss = K.sum(K.square(y_true-y_pred), axis=(1, 2, 3)) # image loss
        kl_loss = 0.5 * K.mean(-z_var - 1 + K.exp(z_var) + z_mean**2, axis=1) #KL divergence loss between N(0, 1) and N(z_mean, z_var)
        return K.mean(im_loss + kl_loss, axis=0)

    model = Model(enc_in, dec_model(z_mean), name='Keras-VAE')
    model.compile(optimizer='adam', loss=vae_loss)

    return model, enc_model, dec_model


def train_model(n=2, hidden_dim=512, epochs=10, batch_size=32):
    """Trains the VAE model

    Params
    ------

    n : int
        The latent-space dimensionality of the autoencoder

    hidden_dim: int
        Dimensionality of the hidden layers

    epochs: int
        Number of epochs to train on

    batch_size: int
        Batch size during training

    Returns
    -------

    trained: 3-tuple
        Contains the encoder-decoder model as (encoder-decoder, encoder, decocer).
    """
    # load the data
    (x_train, x_val), _ = load_data()

    # load the model
    model, enc_model, dec_model = vae_model(n, hidden_dim)
    model.fit(
        x=x_train, y=x_train, 
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, x_val)
    )

    trained = (model, enc_model, dec_model)

    return trained

# utility functions

def load_data():
    """Loads the mnist data and preprocesses it."""
    # load mnist data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # we only need the x_train and x_test data (the images)
    x_train = x_train/255.
    x_train = np.expand_dims(x_train, -1)
    x_val = x_test/255.
    x_val = np.expand_dims(x_val, -1)

    # we'll also return the labels for plotting
    y = np.concatenate([y_train, y_test])

    return (x_train, x_val), y