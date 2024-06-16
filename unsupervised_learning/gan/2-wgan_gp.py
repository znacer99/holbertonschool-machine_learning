#!/usr/bin/env python3
"""
Task 2
"""

import tensorflow as tf
from tensorflow import keras


class WGAN_GP(keras.Model):
    """
    """

    def __init__(self, generator, discriminator,
                 latent_generator, real_examples,
                 batch_size=200, disc_iter=2,
                 learning_rate=0.005, lambda_gp=10):
        super().__init__()
        """
        """
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = 0.3
        self.beta_2 = 0.9

        self.lambda_gp = lambda_gp
        self.dims = self.real_examples.shape
        self.len_dims = tf.size(self.dims)
        self.axis = tf.range(1, self.len_dims, delta=1, dtype='int32')
        self.scal_shape = self.dims.as_list()
        self.scal_shape[0] = self.batch_size
        for i in range(1, self.len_dims):
            self.scal_shape[i] = 1
        self.scal_shape = tf.convert_to_tensor(self.scal_shape)

        # define the generator loss and optimizer:
        self.generator.loss = lambda x: -tf.reduce_mean(x)
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate, beta_1=self.beta_1,
            beta_2=self.beta_2)
        self.generator.compile(optimizer=self.generator.optimizer,
                               loss=self.generator.loss)

        # define the discriminator loss and optimizer:
        self.discriminator.loss = lambda x, y: (
            -tf.reduce_mean(x) + tf.reduce_mean(y)
            )
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1, beta_2=self.beta_2)
        self.discriminator.compile(optimizer=self.discriminator.optimizer,
                                   loss=self.discriminator.loss)

    # generator of real samples of size batch_size
    def get_fake_sample(self, size=None, training=False):
        """
        """
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    # generator of fake samples of size batch_size
    def get_real_sample(self, size=None):
        """
        """
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    # generator of interpolating samples of size batch_size
    def get_interpolated_sample(self, real_sample, fake_sample):
        """
        """
        u = tf.random.uniform(self.scal_shape)
        v = tf.ones(self.scal_shape) - u
        return u * real_sample + v * fake_sample

    # computing the gradient penalty
    def gradient_penalty(self, interpolated_sample):
        """
        """
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_sample)
            pred = self.discriminator(interpolated_sample, training=True)
        grads = gp_tape.gradient(pred, [interpolated_sample])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=self.axis))
        return tf.reduce_mean((norm - 1.0) ** 2)

    # overloading train_step()
    def train_step(self, useless_argument):
        """
        """

        # ---> training of the discriminator
        for _ in range(self.disc_iter):

            with tf.GradientTape() as tape:

                # watch the discriminator's trainable variables
                tape.watch(self.discriminator.trainable_variables)

                # get a real sample
                real_sample = self.get_real_sample()

                # get a fake sample
                fake_sample = self.get_fake_sample(training=True)

                # get the interpolated sample
                interpolated_sample = self.get_interpolated_sample(
                    real_sample, fake_sample)

                # the loss of the discriminator on real & fake samples
                discr_return_on_fake = self.discriminator(fake_sample,
                                                          training=True)
                discr_return_on_real = self.discriminator(real_sample,
                                                          training=True)
                discr_loss = self.discriminator.loss(discr_return_on_real,
                                                     discr_return_on_fake)

                # compute the gradient penalty
                gp = self.gradient_penalty(interpolated_sample)

                # compute the new loss
                new_discr_loss = discr_loss + gp * self.lambda_gp

            # apply gradient descent to the discriminator
            discr_gradient = tape.gradient(
                new_discr_loss, self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(
                zip(discr_gradient, self.discriminator.trainable_variables))

        # ---> training of the generator
        with tf.GradientTape() as tape:
            # watch the discriminator's trainable variables
            tape.watch(self.generator.trainable_variables)

            # get a fake sample
            fake_sample = self.get_fake_sample(training=True)

            # compute the loss of the generator on this fake samples
            discr_return_on_fake = self.discriminator(fake_sample,
                                                      training=True)
            gen_loss = self.generator.loss(discr_return_on_fake)

        # apply gradient descent to the discriminator
        gen_gradient = tape.gradient(gen_loss,
                                     self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables))

        return {"discr_loss": discr_loss, "gen_loss": gen_loss, "gp": gp}
