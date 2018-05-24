"""
MDGAN generates three-dimensional conformations that resemble the ones provided
as training data (MD simulations).
"""
import numpy as np
from keras.layers import Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.models import Sequential
from keras.optimizers import Adam
from utils import make_trainable, make_latent_samples, make_labels
from sklearn.model_selection import train_test_split


class MDGAN():
    def __init__(self, n_atoms, noise_dim=100, gen_lr=2e-4, disc_lr=1e-3,
                 gan_lr=2e-4):
        self.n_atoms = n_atoms
        self.noise_dim = noise_dim
        self.generator = self.build_generator(lr=gen_lr)
        self.discriminator = self.build_discriminator(lr=disc_lr)
        self.gan = self.build_GAN(lr=gan_lr)

    def __repr__(self):
        self.gan.summary()

    def build_generator(self, lr):
        g = Sequential([
            Dense(2 * 2 * self.noise_dim, input_dim=self.noise_dim),
            BatchNormalization(),
            LeakyReLU(0.2),
            Dense(self.n_atoms * 3, input_dim=self.noise_dim * 2 * 2),
            Reshape((self.n_atoms, 3, 1))
        ], name='generator')
        adam = Adam(lr=lr)
        g.compile(adam, loss='binary_crossentropy')
        return g

    def build_discriminator(self, lr):
        d = Sequential([
            Conv2D(32, 3, padding='same', strides=2, input_shape=(self.n_atoms, 3, 1)),
            LeakyReLU(0.2),
            Dropout(0.3),

            Conv2D(64, 3, padding='same', strides=1),
            LeakyReLU(0.2),
            Dropout(0.3),


            Conv2D(128, 3, padding='same', strides=1),
            LeakyReLU(0.2),
            Dropout(0.3),

            Flatten(),
            Dense(1, activation='sigmoid')

        ], name='discriminator')
        adam = Adam(lr=lr)
        d.compile(adam, 'binary_crossentropy')
        return d

    def build_GAN(self, lr):
        gan = Sequential([self.generator, self.discriminator])
        adam = Adam(lr=lr)
        gan.compile(adam, 'binary_crossentropy')
        return gan

    def train(self, data, batch_size=250, num_epochs=25, eval_size=200):
        losses = []
        train, test = train_test_split(data)
        for epoch in range(num_epochs):
            for i in range(len(train) // batch_size):
                # ------------------
                # Train Disciminator
                # ------------------
                make_trainable(self.discriminator, True)
                # Get some real conformations from the train data
                real_confs = train[i * batch_size:(i + 1) * batch_size]
                real_confs = real_confs.reshape(-1, self.n_atoms, 3, 1)

                # Sample high dimensional noise and generate fake conformations
                noise = make_latent_samples(batch_size, self.noise_dim)
                fake_confs = self.generator.predict_on_batch(noise)

                # Label the conformations accordingly
                real_confs_labels, fake_confs_labels = make_labels(batch_size)

                self.discriminator.train_on_batch(real_confs, real_confs_labels)
                self.discriminator.train_on_batch(fake_confs, fake_confs_labels)

                # --------------------------------------------------
                #  Train Generator via GAN (swith off discriminator)
                # --------------------------------------------------
                noise = make_latent_samples(batch_size, self.noise_dim)
                make_trainable(self.discriminator, False)
                g_loss = self.gan.train_on_batch(noise, real_confs_labels)

            # Evaluate performance after epoch
            conf_eval_real = test[np.random.choice(len(test), eval_size, replace=False)]
            conf_eval_real = conf_eval_real.reshape(-1, self.n_atoms, 3, 1)
            noise = make_latent_samples(eval_size, self.noise_dim)
            conf_eval_fake = self.generator.predict_on_batch(noise)

            eval_real_labels, eval_fake_labels = make_labels(eval_size)

            d_loss_r = self.discriminator.test_on_batch(conf_eval_real, eval_real_labels)
            d_loss_f = self.discriminator.test_on_batch(conf_eval_fake, eval_fake_labels)
            d_loss = (d_loss_r + d_loss_f) / 2

            # we want the fake to be realistic!
            g_loss = self.gan.test_on_batch(noise, eval_real_labels)

            print("Epoch: {:>3}/{} Discriminator Loss: {:>6.4f} Generator Loss: {:>6.4f}".format(epoch + 1, num_epochs, d_loss, g_loss))

            losses.append((d_loss, g_loss))
        return losses
