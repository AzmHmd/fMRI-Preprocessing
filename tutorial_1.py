
# https://colab.research.google.com/github/akeshavan/IntroDL/blob/master/IntroToKeras.ipynb#scrollTo=ZCR5NALKsm1o
# *********************************************************************************************************
# 1. Loading Data
# *********************************************************************************************************

'''
!mkdir sample_data
!wget https://s3-ap-southeast-1.amazonaws.com/ohbm2018/sample_data/IXI_small.zip
!unzip IXI_small.zip -d sample_data/
'''

%matplotlib inline
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from keras.utils import to_categorical


def load_image(filename):
    data_slice = plt.imread(filename)
    assert data_slice.shape == (256,256), "image file is not the right shape"
    return data_slice


# *********************************************************************************************************
# 2. Viewing data
# *********************************************************************************************************
# initialize an array that is the shape we need:

data_array = np.zeros((N, 256, 256, 1)) 
# the last dimension is the number of channels in the image.
# a grayscale image has 1 channel, while an RGB image has 3 channels (R,G,B).
# Could be MR modalities (channel1 = T1, channel2 = T2)

# iterate through all of the image files
for i, file in enumerate(image_files):
    # in Python, enumerate is a special function that gives us 
    # an index (i) and the item in the list (file)
    
    
    # load the file using our function
    data_slice = load_image(file)
    
    # put the data into our array, X
    data_array[i, :, :, 0] = data_slice



def view_slice(data, index):
    plt.imshow(data[index,:,:,0],cmap=plt.cm.Greys_r)
    plt.axis('off');


view_slice(data_array, 0)

# *********************************************************************************************************
# 3. Split data into a training and testing set
# *********************************************************************************************************

np.random.seed(0) # set the random number generator seed, for consistency

indices = np.arange(N) # returns a 1D array, [0,1,...N-1]
np.random.shuffle(indices) # shuffles the indicies

# the first 80% of the data will be the training set.
N_80p = int(0.8 * N)
indices_train = indices[:N_80p]
X_train = data_array[indices_train,:,:,:]

# the last 20% of the data will be the testing set.

indices_test = indices[N_80p:]
X_test = data_array[indices_test,:,:,:]

print(X_train.shape, X_test.shape)

# *********************************************************************************************************
# 4. Introducing an Anterior-Posterior Flip
# *********************************************************************************************************
X_train_flip = X_train[:, :, ::-1, :]
X_test_flip = X_test[:, :, ::-1, :]

X_train = np.vstack((X_train, X_train_flip))
X_test = np.vstack((X_test, X_test_flip))

print(X_train.shape, X_test.shape)
# *********************************************************************************************************
# 5. Creating the outcome variable
# *********************************************************************************************************
y_train_label = np.zeros(X_train.shape[0])

N_train_half = int(X_train.shape[0] / 2)
y_train_label[:N_train_half] = 1


y_test_label = np.zeros(X_test.shape[0])

N_test_half = int(X_test.shape[0] / 2)
y_test_label[:N_test_half] = 1


y_train = to_categorical(y_train_label) # to make 2-D data
y_test = to_categorical(y_test_label)
print(y_train)

shuffled_train_indices = np.arange(2*N_train_half)
np.random.shuffle(shuffled_train_indices)

X_train = X_train[shuffled_train_indices, :,:,:]
y_train = y_train[shuffled_train_indices, :]


# *********************************************************************************************************
# 6. Creating a Sequential Model
# *********************************************************************************************************
import tensorflow as tf
from keras.models import Sequential

from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D

from keras.optimizers import Adam, SGD

from keras import backend as K
K.clear_session()


kernel_size = (3, 3)
n_classes = 2

filters = 8

model = Sequential()

model.add(Conv2D(filters, kernel_size, activation='relu', input_shape=(256, 256, 1)))
# zero mean unit variance
model.add(BatchNormalization())

model.add(MaxPooling2D())
model.add(Conv2D(filters*2, kernel_size, activation='relu'))

model.add(MaxPooling2D())
model.add(Conv2D(filters*4, kernel_size, activation='relu'))

model.add(MaxPooling2D())
model.add(Conv2D(filters*8, kernel_size, activation='relu'))

model.add(MaxPooling2D())
model.add(Conv2D(filters*16, kernel_size, activation='relu'))

model.add(MaxPooling2D())
model.add(Conv2D(filters*32, kernel_size, activation='relu'))

model.add(MaxPooling2D())
model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(n_classes, activation='softmax'))

learning_rate = 1e-5
# optimizer
adam = Adam(lr=learning_rate)
sgd = SGD(lr=learning_rate)


model.compile(loss='categorical_crossentropy',
                 optimizer=adam, # swap out for sgd 
                 metrics=['accuracy'])

model.summary()


# *********************************************************************************************************
# 7. Fitting the Model
# *********************************************************************************************************
fit = model.fit(X_train, y_train, epochs=5, batch_size=2)


def get_figure():
    """
    Returns figure and axis objects to plot on. 
    Removes top and right border and ticks, because those are ugly
    """
    fig, ax = plt.subplots(1)
    plt.tick_params(top='off', right='off', which='both') 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return fig, ax

fig, ax = get_figure()

epoch = np.arange(5) + 1
ax.plot(epoch, fit.history['acc'], marker="o", linewidth=2, color="steelblue", label="accuracy")
ax.plot(epoch, fit.history['loss'], marker="o", linewidth=2, color="orange", label="loss")
ax.set_xlabel('epoch')
ax.legend(frameon=False);

# *********************************************************************************************************
# 8. Evaluating the model
# *********************************************************************************************************
model.evaluate(X_test, y_test)

# WHAT DID IT MESS UP?
y_pred = model.predict(X_test)
fig, ax = get_figure()
ax.hist(y_pred[:,0], color="steelblue")
np.nonzero(np.isclose(y_test[:,0], y_pred[:,0], atol=4e-1) == False)
view_slice(X_test, 47) # view the messed up slices

# What about the training set?
y_train_pred = model.predict(X_train)
fig, ax = get_figure()
ax.hist(y_train_pred[:,0], color="steelblue")
np.nonzero(np.isclose(y_train[:,0], y_train_pred[:,0], atol=2e-1) == False)
view_slice(X_train, 793) # view the messed up slices

'''

Data Input

    We loaded 2D JPG images --> numpy array. You can use the nibabel library to load nifti images (you'd need a GPU)
    We created 2 classes of images that were perfectly balanced: one oriented "correctly" and the other oriented "incorrectly". You can try to train a model to predict age or sex from the IXI dataset demographics spreadsheet
    Our dependent variable y_train, y_test was a binary label, but we converted it to a 2D array

Training and Testing

    it is not wise to train on your entire dataset, because you will overfit.
    ALWAYS split your data

Model Creation

    We used a Sequential model, but you can also use the more generic Model class
    model.summary() is very useful
    try fiddling with learning rate and optimizers (Adam, SGD) -- what happens?
    try adding or removing layers -- what happens?

Output

    model.evaluate returns the loss, accuracy of your model
    model.predict returns an array with the predicted values

Documentation

    There is excellent documentation online! Check out https://keras.io to get started



'''

# *********************************************************************************************************
# 10. Extra: Visualizing Activations
# *********************************************************************************************************
layer_dict = dict([(layer.name, layer) for layer in model.layers])

def show_activation(layer_name):
    
    layer_output = layer_dict[layer_name].output

    fn = K.function([model.input], [layer_output])
    
    inp = X_train[0:1]
    
    this_hidden = fn([inp])[0]
    
    # plot the activations from the first 8 filters
    plt.figure(figsize=(15,8))
    for i in range(8):
        plt.subplot(1,8,i+1)
        plt.imshow(this_hidden[0,:,:,i], plt.cm.Greys_r)
        plt.axis('off')
    
    return 

show_activation('conv2d_1')