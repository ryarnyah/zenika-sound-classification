import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Concatenate
from tensorflow.keras.layers import Input, Dense, TimeDistributed, LSTM, Dropout, Activation
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import ELU
import numpy as np

from matplotlib import pyplot as plt
plt.switch_backend('agg')

import librosa.display

import librosa

PLOT_WIDTH = 4.8
PLOT_HEIGHT = 4.8

class DelegateDataGenerator(keras.utils.Sequence):

    def __init__(self, generators = []):
        self.generators = generators
        self.on_epoch_end()

    def __len__(self):
        size = 0
        for g in self.generators:
            size += g.__len__()
        return size

    def __get_generator(self, index):
        i = 0
        for g in self.generators:
            if index < (i + len(g)):
                return g, index - i
            i += len(g)
        return None, None

    def on_epoch_end(self):
        self.indexes = list(range(len(self)))
        np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        i = self.indexes[index]
        generator, current_index = self.__get_generator(i)
        return generator.__getitem__(current_index)

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, filename, y_value = 1, time_used = 0.05):
        'Initialization'
        file_data = librosa.load(filename, sr=44100)
        self.frequency = file_data[1]
        self.input_data = file_data[0]
        self.data_len = len(self.input_data)
        self.batch_size = int(np.floor(self.frequency * time_used))
        self.y_value = y_value

    def set_data_len(self, data_len):
        self.data_len = data_len

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.data_len / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        data = self.input_data[index*self.batch_size:(index+1)*self.batch_size]

        #print('[%d:%d:%d:%d]' % (index, self.batch_size, len(self.input_data), len(data)))
        D_highres = librosa.stft(data, hop_length=256, n_fft=4096)
        melgram = librosa.amplitude_to_db(np.abs(D_highres), ref=np.max)
        #melgram = librosa.amplitude_to_db(
        #    librosa.feature.melspectrogram(data, sr=self.frequency, n_mels=96))

        fig = plt.figure(figsize=[PLOT_WIDTH, PLOT_HEIGHT])
        try:
            ax1 = fig.add_subplot(111)
            img = librosa.display.specshow(melgram, hop_length=256, x_axis='time', y_axis='log',ax=ax1)
            #ax1.plot(data)

            canvas = ax1.figure.canvas
            canvas.draw()

            x = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
            x = x.reshape((1,) + fig.canvas.get_width_height()[::-1] + (3,))
            y = np.array([self.y_value])
        finally:
            plt.close(fig)

        return x, y


training_generator = DataGenerator('valid_normal.wav', 1)
bad_training_generator = DataGenerator('valid_euhh.wav', 0)

test_generator = DelegateDataGenerator([training_generator, bad_training_generator])

valid_training_generator = DataGenerator('normal.wav', 1)
valid_bad_training_generator = DataGenerator('euhh.wav', 0)

valid_test_generator = DelegateDataGenerator([valid_training_generator, valid_bad_training_generator])

#base_model = keras.applications.Xception(
#    weights='imagenet',  # Load weights pre-trained on ImageNet.
#    input_shape=(int(PLOT_WIDTH * 100), int(PLOT_HEIGHT * 100), 3),
#    include_top=False
#)
#base_model.trainable = False
#
#inputs = keras.Input(shape=(int(PLOT_WIDTH * 100), int(PLOT_HEIGHT * 100), 3))
#x = base_model(inputs, training=False)
## Convert features of shape `base_model.output_shape[1:]` to vectors
#x = keras.layers.GlobalAveragePooling2D()(x)
## A Dense classifier with a single unit (binary classification)
#outputs = keras.layers.Dense(1, activation='sigmoid')(x)
#model = keras.Model(inputs, outputs)

####
nb_layers = 3
pool_size = (2, 2)
nb_filters = 64
kernel_size = (3, 3)

model = Sequential()
model.add(Convolution2D(nb_filters, kernel_size, input_shape=(int(PLOT_WIDTH * 100), int(PLOT_HEIGHT * 100), 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))

for layer in range(nb_layers-1):
    model.add(Convolution2D(nb_filters, kernel_size))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))

model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[keras.metrics.BinaryAccuracy()])

print(model.summary())

model.fit(x=test_generator,
#          validation_data=valid_test_generator,
          use_multiprocessing=False,
          workers=4,
          epochs=20,
)
result = model.evaluate(valid_test_generator)
print(dict(zip(model.metrics_names, result)))

model.save("test2.h2")
