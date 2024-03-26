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

# Fitting the CNN Model to Images
# Image Augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

training_set = train_datagen.flow_from_directory(
    "dataset/training_set/", target_size=(64, 64), batch_size=32, class_mode="binary"
)

test_set = test_datagen.flow_from_directory(
    "dataset/test_set/", target_size=(64, 64), batch_size=32, class_mode="binary"
)
