# importing the packages
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing import ImageDataGenerator

# Initialising the CNN
classifier = Sequential()

# Step 1 - Covolution
classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation="relu"))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adding the second convolution layer
classifier.add(Convolution2D(32, 3, 3, activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full Connection
classifier.add(Dense(units=128, activation="relu"))

# add the output layer
classifier.add(Dense(units=1, activation="sigmoid"))

# compile the model
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
