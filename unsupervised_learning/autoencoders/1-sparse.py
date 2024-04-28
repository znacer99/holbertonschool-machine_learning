#!/usr/bin/env python3
"""
Script that creates a sparse autoencoder
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    Returns: encoder, decoder, auto
            encoder is the encoder model
            decoder is the decoder model
            auto is the full autoencoder model
    """
    input = keras.Input
    dense = keras.layers.Dense
    model = keras.Model

    inputs = input((input_dims,))
    encoded = dense(hidden_layers[0], activation='relu')(inputs)
    for i in range(1, len(hidden_layers)):
        encoded = dense(hidden_layers[i], activation='relu')(encoded)

    regulizer = keras.regularizers.l1(lambtha)
    encoded = dense(latent_dims,
                    activation='relu',
                    activity_regularizer=regulizer)(encoded)

    encoded_img = input((latent_dims,))

    encoder = model(inputs, encoded)

    decoded = encoded_img
    for h in reversed(hidden_layers):
        decoded = dense(h, activation='relu')(decoded)

    decoded = dense(input_dims, activation='sigmoid')(decoded)

    decoder = model(encoded_img, decoded)
    out_decoder = decoder(encoder(inputs))
    auto = model(inputs, out_decoder)
    auto.compile(loss='binary_crossentropy', optimizer='adam')

    return encoder, decoder, auto
