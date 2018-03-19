import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import keras
import sklearn

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Cropping2D
from keras.models import load_model


# Check tensorflow is the backend
print(keras.backend.backend())
keras.backend.image_dim_ordering()


# Training Data

samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Training Data Generator

def generator(samples, batch_size=32, directory="data"):
    """
    Based on generator from Udacity class: Generators
    
    samples: full set of training data
    batch_size: size of training batch
    directory: directory location training data is saved
    
    """
    
    num_samples = len(samples)
    
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            
            for batch_sample in batch_samples:
                
                filename = directory+'/IMG/' + batch_sample[0].split('/')[-1]
                
                image = cv2.imread(filename)
                images.append(image)
                
                measurement = float(batch_sample[3])
                measurements.append(measurement)
                
                # Image Augmentation (Flip)
                images.append(cv2.flip(image, 1))
                measurements.append(measurement * -1.0)
                 

            X_train = np.array(images)
            y_train = np.array(measurements)
            
            yield (X_train, y_train)


            
# Generate Base Training and Validation Data

train_generator = generator(train_samples, batch_size=32, directory="data")
validation_generator = generator(validation_samples, batch_size=32, directory="data")


# Model (NVIDIA) Architecture

model = Sequential()

# Normalisation Layer
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))

# Cropping Layer
model.add(Cropping2D(cropping=((70, 25), (0,0))))

# Convolution Layers
model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
model.add(Dropout(0.5))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
model.add(Dropout(0.5))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))

# Fully Connected Layers
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))

# Output Layer
model.add(Dense(1))


# Model Training

model.compile(loss="mse", optimizer="adam")

history = model.fit_generator(train_generator, samples_per_epoch=2*len(train_samples),
                             validation_data=validation_generator, nb_val_samples=2*len(validation_samples), 
                              nb_epoch=5)
model.save("model.h5")


# Plot the training and validation loss for each epoch

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model mean squared error loss")
plt.ylabel("mean squared error loss")
plt.xlabel("epoch")
plt.legend(["training set", "validation set"], loc="upper right")
plt.show()




# Retraining the Model

samples = []

with open('extra_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
                
train_samples, validation_samples = train_test_split(samples, test_size=0.2)



# Generate Extra Training and Validation Data

train_generator = generator(train_samples, batch_size=32, directory="extra_data")
validation_generator = generator(validation_samples, batch_size=32, directory="extra_data")


# Remove existing model, and load previous model

del model
model = load_model("model.h5")


# Model Re-Training

model.compile(loss="mse", optimizer="adam")
history = model.fit_generator(train_generator, samples_per_epoch=2*len(train_samples),
                             validation_data=validation_generator, nb_val_samples=2*len(validation_samples), 
                              nb_epoch=2)
model.save("new_model.h5")


# Plot the training and validation loss for each epoch

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model mean squared error loss")
plt.ylabel("mean squared error loss")
plt.xlabel("epoch")
plt.legend(["training set", "validation set"], loc="upper right")
plt.show()
