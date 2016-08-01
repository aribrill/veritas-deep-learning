"""
train.py

Load and parse VERITAS data and use to train a deep learning model for
classification as gamma ray signal or hadronic background.
"""

import numpy as np

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Merge, Flatten
from keras.optimizers import Adam

from process_data import load_data

# Load the VERITAS data and parse into usable format
print "Loading data..."
data, labels = load_data(1000, '59521_data.txt', '59521_gammas.txt')
print data.shape
print labels.shape

# Set hyperparameters
lr = 0.0001

# Set up the model
model = Sequential()
model.add(Convolution2D(64, 6, 6, activation='relu', border_mode='same', input_shape=(4, 64, 64)))
model.add(Convolution2D(64, 6, 6, activation='relu', border_mode='same'))
model.add(Convolution2D(64, 6, 6, activation='relu', border_mode='same'))
model.add(Convolution2D(64, 6, 6, activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))
model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))
model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))
model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(1024, 3, 3, activation='relu', border_mode='same'))
model.add(Convolution2D(1024, 3, 3, activation='relu', border_mode='same'))
model.add(Convolution2D(1024, 3, 3, activation='relu', border_mode='same'))
model.add(Convolution2D(1024, 3, 3, activation='relu', border_mode='valid'))
model.add(Convolution2D(1024, 3, 3, activation='relu', border_mode='valid'))
model.add(Convolution2D(1, 4, 4, activation='sigmoid', border_mode='valid'))
model.add(Flatten())

model.compile(loss='binary_crossentropy', optimizer=Adam(lr=lr),
        metrics=['accuracy'])

# Train the model
#histories = []
hist = model.fit(data, labels, nb_epoch=1, validation_split=0.)
#histories.append([lr, hist])

# Evaluate the model
#loss_and_metrics = model.evaluate(test_data, test_labels, batch_size=32)

# Save the trained model
open('trained_model_architecture.json', 'w').write(model.to_json())
model.save_weights('trained_model_weights.h5', overwrite=True)

#for hist in sorted(histories, key=lambda hist: hist[2].history['val_acc'][-1]):
#    print "LR:", hist[0], "RS:", hist[1]
#    print ">>>", "ACC:", hist[2].history['acc'][-1], "LOSS:",\
#            hist[2].history['loss'][-1]
#    print ">>>", "VAL ACC:", hist[2].history['val_acc'][-1], "VAL LOSS:",\
#            hist[2].history['val_loss'][-1]



