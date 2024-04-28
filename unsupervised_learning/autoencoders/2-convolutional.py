#!/usr/bin/env python3
"""
Script that creates a convolutional autoencoder
"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Returns: encoder, decoder, auto
            encoder is the encoder model
            decoder is the decoder model
            auto is the full autoencoder model
    """
    input = keras.Input
    model = keras.Model
    Conv2D = keras.layers.Conv2D

    inputs = input(input_dims)
    encoded = inputs
    for f in filters:
        encoded = Conv2D(f, (3, 3), activation='relu', padding='same')(encoded)
        encoded = keras.layers.MaxPooling2D((2, 2), padding='same')(encoded)

    encoded_img = input(latent_dims)
    decoded = encoded_img
    for i in reversed(range(1, len(filters))):
        f = filters[i]
        decoded = Conv2D(f, (3, 3), activation='relu', padding='same')(decoded)
        decoded = keras.layers.UpSampling2D((2, 2))(decoded)

    decoded = Conv2D(filters[0],
                     (3, 3),
                     activation='relu',
                     padding='valid')(decoded)
    decoded = keras.layers.UpSampling2D((2, 2))(decoded)
    decoded = Conv2D(input_dims[-1], (3, 3),
                     activation='sigmoid', padding='same')(decoded)

    encoder = model(inputs, encoded)
    decoder = model(encoded_img, decoded)

    out_decoder = decoder(encoder(inputs))
    auto = model(inputs, out_decoder)
    auto.compile(optimizer="adam", loss="binary_crossentropy")
    return encoder, decoder, auto
