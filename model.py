import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import keras

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

lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
images = []
measurements = []

for line in lines[1:]:
    
    source_path = line[0]
    filename = source_path.split("/")[-1]
    current_path = 'data/IMG/' + filename
    
    image = cv2.imread(current_path)
    images.append(image)

    measurement = float(line[3])
    measurements.append(measurement)

print(len(images), len(measurements))

# Image Augmentation 

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    
    # Flip each image and measurement
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement * -1.0)
    
# Training Data Setup

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)



# Model (NVIDIA) Architecture
model = Sequential()

# Normalisation Layer
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))

# Cropping Layer
model.add(Cropping2D(cropping=((70, 25), (0,0))))

# Convolution Layers
model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
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
history = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save("model.h5")


# Plot the training and validation loss for each epoch

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model mean squared error loss")
plt.ylabel("mean squared error loss")
plt.xlabel("epoch")
plt.legend(["training set", "validation set"], loc="upper right")
plt.show()



# Extra Training Data

images = []
measurements = []
lines = []

with open('extra_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
                
for line in lines[1:]:
    source_path = line[0]
    filename = source_path.split("/")[-1]
    current_path = 'extra_data/IMG/' + filename
    
    image = cv2.imread(current_path)
    images.append(image)
    
    measurement = float(line[3])
    measurements.append(measurement)

# Image Augmentation 

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    
    # Flip each image and measurement
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement * -1.0)

    
# Training Data Setup

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)


# Model Re-Training

del model
model = load_model("model.h5")

model.compile(loss="mse", optimizer="adam")
history = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)

model.save("new_model.h5")
