#!/usr/bin/env python3

"""
Function that creates a variational autoencoder
"""
import tensorflow.keras as keras


def sampling(args):
    """Sample from variational space for output"""
    mean, logvar = args
    batch = keras.backend.shape(mean)[0]
    dim = keras.backend.int_shape(mean)[1]
    epsilon = keras.backend.random_normal(shape=(batch, dim))
    return mean + keras.backend.exp(0.5 * logvar) * epsilon


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Returns: encoder, decoder, auto
        @encoder is the encoder model
        @decoder is the decoder model
        @auto is the full autoencoder model
    """
    Input = keras.Input
    Dense = keras.layers.Dense
    Lambda = keras.layers.Lambda
    Model = keras.Model
    K = keras.backend

    inputs = Input((input_dims,))
    h = inputs
    for hidden_layer in hidden_layers:
        h = Dense(hidden_layer, activation="relu")(h)
    z_mean = Dense(latent_dims, activation=None)(h)
    z_log_sigma = Dense(latent_dims, activation=None)(h)

    z = Lambda(sampling)([z_mean, z_log_sigma])

    encoder = Model(inputs, [z_mean, z_log_sigma, z])

    latent_inputs = Input((latent_dims,))

    x = latent_inputs
    for hidden_layer in reversed(hidden_layers):
        x = Dense(hidden_layer, activation='relu')(x)
    outputs = Dense(input_dims, activation='sigmoid')(x)
    decoder = Model(latent_inputs, outputs)

    outputs = decoder(encoder(inputs))
    vae = Model(inputs, outputs)

    def kl_reconstruction_loss(y_true, y_pred):
        reconstruction_loss = keras.losses.binary_crossentropy(inputs, outputs)
        reconstruction_loss *= input_dims
        kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        return vae_loss

    vae.compile(optimizer="adam", loss=kl_reconstruction_loss)
    return encoder, decoder, vae
