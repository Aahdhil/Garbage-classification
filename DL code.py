''' Importing necessary libraries:os for file handling, numpy for arrays and dataset handling, matplotlib
for plotting graphs, tensorflow for building and training models, convo2d for applying convolutional
operation, flatten converts 2d feature maps into 1d vector and dropout to prevent overfitting'''
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Setting up the paths for training and testing data
train_path = 'dataset/train'
test_path = 'dataset/test'

# Data preprocessing such as resizing images, rescaling and having a batch size for processing images
image_size = (150, 150)
batch_size = 32
datagen = ImageDataGenerator(rescale=1./255)

# load images from folders into batches by assigning labels and resizing
train_data = datagen.flow_from_directory(
    train_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_data = datagen.flow_from_directory(
    test_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

#Heart of the CNN model architeture
''' It starts building a sequential model where each time increasing the filters each time allows the model
to learn more detailed features. Below are the 3 convolutions and relu activation function. Relu is used
for simple yet efficent method for training the model. There is a dropout function to prevent overfitting
and generalizing the nodes'''
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_data.num_classes, activation='softmax') 
])

#Compile the Model
'''The Adam optimizer updates model weights using gradient descent. It's fast and adapts learning rates
on its own (a mix of RMSprop + Momentum). Very popular and effective. Categorical crossentropy is
used for multi-class classification problems (here there are 6 types of garbage).'''
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Training the model
''' Epochs is the number of times it has trained on the train data. After every epoch, the model evaluvates
itself on the test data to keep track of overfitting.'''
history = model.fit(
    train_data,
    epochs=15,
    validation_data=test_data
)

#Predicting the type of garbage when the image is given after training and testing itself
class_names = list(train_data.class_indices.keys())
test_folder = 'dataset/test_images' 

# === Loop over each image in the folder ===
for filename in os.listdir(test_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(test_folder, filename)
        
        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(150, 150))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]

        # Display image and prediction
        plt.imshow(img)
        plt.title(f"{filename} → Predicted: {predicted_class}")
        plt.axis('off')
        plt.show()

# Plot Accuracy and Loss
plt.figure(figsize=(12,5))

''' Training accuracy is based on training data and val/validation accuracy is based on testing data.
Training the model just memeorizes the answers whereas validating helps us to see if the model is really
learning.'''
# Plot Accuracy of the model
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Val Accuracy', marker='x')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

#Plot Loss value 
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Val Loss', marker='x')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
