"""
train.py

Load and parse VERITAS data and use to train a deep learning model for
classification as gamma ray signal or hadronic background.
"""

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.regularizers import l2

# Load the VERITAS data and parse into usable format
# Use meaningless random fake data for now
n_train = 100
data = np.random.random((n_train, 499))
labels = np.random.randint(2, size=(n_train,1))

# Set up the model
reg = 0.0 # regularization
model = Sequential()
model.add(Dense(32, input_dim=499, W_regularizer=l2(reg)))
model.add(Activation('relu'))
model.add(Dense(1, W_regularizer=l2(reg)))
model.add(Activation('relu'))
model.compile(loss='mean_squared_error', optimizer=SGD(lr=1e-2), metrics=['accuracy'])

## Train the model
model.fit(data, labels, nb_epoch=5)

# Save the trained model
open('trained_model_architecture.json', 'w').write(model.to_json())
model.save_weights('trained_model_weights.h5', overwrite=True)
