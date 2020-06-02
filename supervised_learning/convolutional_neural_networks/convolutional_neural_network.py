# Convolutional Neural Network

# Part 1: Building the CNN


# importing keras packages and libraries
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# initialize CNN
classifier = Sequential()

# Step 1: Convolution
classifier.add(Convolution2D(32, #32 feature detectors ie number of layers
                            3, #no of rows of feature detectors
                            3, #no of cols of feature detectors
                            input_shape = (64,64,3), #3 channels with res 64 x 64
                            activation = "relu")) #to avoid linearity

# Step 2: Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2), dim_ordering="tf"))

# adding another convulational layer
classifier.add(Convolution2D(32,
                            3,
                            3,
                            activation = "relu"))
classifier.add(MaxPooling2D(pool_size = (2, 2), dim_ordering="tf"))
# Step 3: Flattening
classifier.add(Flatten())

# Step 4: Full connection
classifier.add(Dense(output_dim = 128,
                     activation = 'relu'))
# making output layer
classifier.add(Dense(output_dim = 1,
                     activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = "adam",
                   loss = "binary_crossentropy",
                   metrics = ["accuracy"])

# Part 2: Fitting the CNN to images
from keras.preprocessing.image import ImageDataGenerator

#image augmentation
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')
classifier.fit(training_set,
                steps_per_epoch=8000,
                epochs=25,
                validation_data=test_set,
                validation_steps=2000)

