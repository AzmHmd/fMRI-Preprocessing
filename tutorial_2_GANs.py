from __future__ import print_function, division
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import sys
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline



# *********************************************************************************
# Building the generator
# *********************************************************************************
img_shape = (28,28,1)

def build_generator(z_dim):
    model = Sequential()
    model.add(Dense(256, input_dim=z_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.prod(img_shape), activation='tanh'))
    model.add(Reshape(img_shape))
    
    model.summary()
    noise = Input(shape=(z_dim,))
    img = model(noise)
    
    return Model(noise, img)



# *********************************************************************************
# Building the discriminator
# *********************************************************************************
def build_discriminator():

    model = Sequential()

    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    # (!!!) No softmax
    model.add(Dense(1))
    model.summary()

    img = Input(shape=img_shape)
    validity = model(img)

    return Model(img, validity)

# *********************************************************************************
# Load the dataset
# *********************************************************************************

batch_size = 32

# Load the dataset
(X_train, _), (_, _) = mnist.load_data()

# Rescale -1 to 1
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
X_train = np.expand_dims(X_train, axis=3)

# Adversarial ground truths
real_label = np.ones((batch_size, 1))
fake_label = np.zeros((batch_size, 1))

idxs = np.arange(0, len(X_train))
np.random.shuffle(idxs)
c = 0
plt.figure(figsize=(8,4))
for i in range(4):
  for j in range(4):
    plt.subplot(4,4,c+1)
    plt.imshow(X_train[idxs[c],:,:,0], cmap="gray")
    plt.axis('off')
    c += 1
# *********************************************************************************
# Understanding the loss function
# *********************************************************************************
# Specify the optimiser. Adam is a pretty
# safe choice these days.
optimizer = Adam(0.0002, 0.5)
# Build and compile the discriminator
discriminator = build_discriminator()
# Print a textual description of the network.
discriminator.summary()
# Compile.
discriminator.compile(loss='mse',optimizer=optimizer,metrics=['accuracy'])

# *********************************************************************************
# Build the generator
z_dim = 100
generator = build_generator(z_dim)

generator.summary()

# The generator takes noise as input and generated imgs
z = Input(shape=(z_dim,))
img = generator(z)

# For the combined model we will only train the generator
discriminator.trainable = False

# The valid takes generated images as input and determines validity
disc_out = discriminator(img)

# The combined model  (stacked generator and discriminator)
# Trains generator to fool discriminator
combined = Model(z, disc_out)
# (!!!) Optimize w.r.t. MSE loss instead of crossentropy
combined.compile(loss='mse', optimizer=optimizer)


# *********************************************************************************
# Training the model
# *********************************************************************************

# The number of iterations.
iters = 20000

for iter_ in range(iters):

    # ---------------------
    #  Train Discriminator
    # ---------------------

    # Select a random batch of images
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    imgs = X_train[idx]

    # Sample noise as generator input
    noise = np.random.normal(0, 1, (batch_size, 100))

    # Generate a batch of new images
    gen_imgs = generator.predict(noise)

    # Train the discriminator
    d_loss_real = discriminator.train_on_batch(imgs, real_label)
    d_loss_fake = discriminator.train_on_batch(gen_imgs, fake_label)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


    # ---------------------
    #  Train Generator
    # ---------------------

    g_loss = combined.train_on_batch(noise, real_label)

    # Plot the progress
    if iter_ % 300 == 0:
      print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (iter_, d_loss[0], 100*d_loss[1], g_loss))


# *********************************************************************************
# Generating samples
# *********************************************************************************

def sample_images():
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, 100))
    gen_imgs = generator.predict(noise)

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1

​