# reference to : https://colab.research.google.com/github/akeshavan/IntroDL/blob/master/IntroToKeras.ipynb#scrollTo=ZCR5NALKsm1o
# *********************************************************************************************************
# 0. Required libraries
# *********************************************************************************************************
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D
from keras.optimizers import Adam, SGD
from keras import backend as K

# *********************************************************************************************************
# 1. Required functions
# *********************************************************************************************************
def load_image(filename):
    data_slice = plt.imread(filename)
    assert data_slice.shape == (256,256), "image file is not the right shape"
    return data_slice

def view_slice(data, index):
    plt.imshow(data[index,:,:,0],cmap=plt.cm.Greys_r)
    plt.axis('off');
    plt.savefig('/results/viewSlice_'+str(index)+'.png')
    
def get_figure():
    fig, ax = plt.subplots(1)
    plt.tick_params(top='off', right='off', which='both') 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return fig, ax   
    
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
    plt.savefig('/results/show_activation_'+layer_name+'.png')
    return
# *********************************************************************************************************
# 2. Loading and viewing data
# *********************************************************************************************************
# initialize an array that is the required shape:
data_array = np.zeros((N, 256, 256, 1)) 

image_files = glob('dataset/*.jpg')
N = len(image_files)
print(N)

# iterate through all of the image files
for i, file in enumerate(image_files):
    
    data_slice = load_image(file)
    data_array[i, :, :, 0] = data_slice

view_slice(data_array, 0)

# *********************************************************************************************************
# 3. Split data into a training and testing set
# *********************************************************************************************************
np.random.seed(0) 
indices = np.arange(N) 
np.random.shuffle(indices) 

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

shuffled_train_indices = np.arange(2*N_train_half)
np.random.shuffle(shuffled_train_indices)

X_train = X_train[shuffled_train_indices, :,:,:]
y_train = y_train[shuffled_train_indices, :]
# *********************************************************************************************************
# 6. Creating a Sequential Model
# *********************************************************************************************************
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

fig, ax = get_figure()
epoch = np.arange(5) + 1
ax.plot(epoch, fit.history['acc'], marker="o", linewidth=2, color="steelblue", label="accuracy")
ax.plot(epoch, fit.history['loss'], marker="o", linewidth=2, color="orange", label="loss")
ax.set_xlabel('epoch')
ax.legend(frameon=False);
plt.savefig('/results/training_process.png')
# *********************************************************************************************************
# 8. Visualizing middle levels
# *********************************************************************************************************
layer_dict = dict([(layer.name, layer) for layer in model.layers])
show_activation('conv2d_1')
show_activation('conv2d_2')
show_activation('conv2d_3')
show_activation('conv2d_4')
show_activation('conv2d_5')

# END ****************
