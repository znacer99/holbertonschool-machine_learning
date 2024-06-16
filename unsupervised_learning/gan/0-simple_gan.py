#!/usr/bin/env python3
"""
Task 0
"""
import tensorflow as tf
from tensorflow import keras


class Simple_GAN(keras.Model):
    """
    """
    def __init__(self, generator, discriminator,
                 latent_generator, real_examples,
                 batch_size=200, disc_iter=2, learning_rate=0.005):
        super().__init__()
        """
        """
        self.latent_generator = latent_generator
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter
        self.real_examples = real_examples

        self.learning_rate = learning_rate
        self.beta_1 = 0.5
        self.beta_2 = 0.9

        # define the generator loss and optimizer:
        self.generator.loss = lambda x: (
            tf.keras.losses.MeanSquaredError()(x, tf.ones(x.shape)))
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1, beta_2=self.beta_2)
        self.generator.compile(
            optimizer=self.generator.optimizer, loss=self.generator.loss)

        # define the discriminator loss and optimizer:
        self.discriminator.loss = lambda x, y: (
            tf.keras.losses.MeanSquaredError()(x, tf.ones(x.shape)) +
            tf.keras.losses.MeanSquaredError()(y, -1 * tf.ones(y.shape)))
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1, beta_2=self.beta_2)
        self.discriminator.compile(
            optimizer=self.discriminator.optimizer,
            loss=self.discriminator.loss)
        self.discr_return_on_real = None
        self.discr_return_on_fake = None
        self.discr_return_on_fake_g = None

    # generator of real samples of size batch_size
    def get_fake_sample(self, size=None, training=False):
        """
        """
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size),
                              training=training)

    # generator of fake samples of size batch_size
    def get_real_sample(self, size=None):
        """
        """
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    # overloading train_step()
    def train_step(self, useless_argument, show=False):
        """
        """
        # --> training of the discriminator
        for _ in range(self.disc_iter):
            with tf.GradientTape() as tape:
                # watch the discriminator's trainable variables
                tape.watch(self.trainable_variables)

                # get a real sample:
                real_sample = self.get_real_sample()

                # get a fake sample:
                fake_sample = self.get_fake_sample(training=True)

                # the loss of the discriminator on real and fake samples
                self.discr_return_on_fake = self.discriminator(fake_sample,
                                                               training=True)
                self.discr_return_on_real = self.discriminator(real_sample,
                                                               training=True)
                discr_loss = self.discriminator.loss(
                    self.discr_return_on_real, self.discr_return_on_fake)

            # apply gradient descent to the discriminator
            discr_gradient = tape.gradient(
                discr_loss, self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(
                zip(discr_gradient, self.discriminator.trainable_variables))

        # ---> training of the generator
        with tf.GradientTape() as tape:
            # watch the discriminator's trainable variables
            tape.watch(self.generator.trainable_variables)

            # get a fake sample:
            fake_sample = self.get_fake_sample(training=True)

            # compute the loss of the generator on this fake samples
            self.discr_return_on_fake_g = self.discriminator(fake_sample,
                                                             training=True)
            gen_loss = self.generator.loss(self.discr_return_on_fake_g)

        # apply gradient descent to the discriminator
        gen_gradient = tape.gradient(gen_loss,
                                     self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables))

        return {"discr_loss": discr_loss, "gen_loss": gen_loss}
