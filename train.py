# from google.colab import drive
# drive.mount('/content/drive')

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Directory where your dataset is stored
data_dir = r'C:\Users\chinm\Downloads\Indian Medicinal Leaves Image Datasets\Medicinal Leaf dataset'

# Define the image dimensions and batch size 
img_height, img_width = 224, 224
batch_size = 32

# Load the pre-trained VGG16 model (without the top layers)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Freeze the pre-trained layers so they are not trainable
for layer in base_model.layers:
    layer.trainable = False

# Define the number of classes in your dataset
num_classes = 80  # Updated to match local dataset

# Build the model
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')  # Use the actual number of classes
])

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Use ImageDataGenerator for data augmentation and preprocessing
train_data_gen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Splitting the data into training and validation sets
)

train_generator = train_data_gen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_data_gen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Define early stopping criteria
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model with early stopping
epochs = 25
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator ,
    callbacks=[early_stopping]   #Add the early stopping callback here
)

# Plot training accuracy and validation accuracy over epochs
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()


# Plot training accuracy and validation accuracy over epochs
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()



import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Directory where your dataset is stored
data_dir = r'C:\Users\chinm\Downloads\Indian Medicinal Leaves Image Datasets\Medicinal Leaf dataset'

# Define the image dimensions and batch size
img_height, img_width = 224, 224
batch_size = 32

# Define the number of classes in your dataset
num_classes = 80  # Updated to match local dataset


# Define the model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')  # Use the actual number of classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Use ImageDataGenerator for data augmentation and preprocessing
train_data_gen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Splitting the data into training and validation sets
)

train_generator = train_data_gen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_data_gen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Train the model
epochs = 25
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
)




# Plot training accuracy and validation accuracy over epochs
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Save the model
model.save('leaf_identification_VGG16.keras')

import os

data_dir = r'C:\Users\chinm\Downloads\Indian Medicinal Leaves Image Datasets\Medicinal Leaf dataset'

# Get a list of subfolder names
subfolder_names = sorted(os.listdir(data_dir))

# Create a dictionary with the format {0: 'subfolder1', 1: 'subfolder2', ...}
subfolder_dict = {i: subfolder_name for i, subfolder_name in enumerate(subfolder_names)}

print(subfolder_dict)


model = load_model('/content/leaf_identification_VGG16.keras')

from google.colab.patches import cv2_imshow
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image as PILImage
import matplotlib.pyplot as plt
from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode
import tensorflow as tf


# Load the trained model
model = load_model('/content/leaf_identification_VGG16.keras')

# Function to capture a photo using the webcam
def take_photo(filename='leaf_image.jpg', quality=0.8):
    js = Javascript('''
        async function takePhoto(quality) {
            const div = document.createElement('div');
            const capture = document.createElement('button');
            capture.textContent = 'Capture';
            div.appendChild(capture);

            const video = document.createElement('video');
            video.style.display = 'block';
            const stream = await navigator.mediaDevices.getUserMedia({ 'video': true });

            document.body.appendChild(div);
            div.appendChild(video);
            video.srcObject = stream;
            await video.play();

            // Resize the output to fit the video element.
            google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

            // Wait for Capture to be clicked.
            await new Promise((resolve) => capture.onclick = resolve);

            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);
            stream.getVideoTracks()[0].stop();
            div.remove();
            return canvas.toDataURL('image/jpeg', quality);
        }
    ''')
    display(js)

    data = eval_js('takePhoto({})'.format(quality))
    binary = b64decode(data.split(',')[1])
    with open(filename, 'wb') as f:
        f.write(binary)
    return filename

# Capture a leaf image
leaf_image_path = 'leaf_image.jpg'  # Update with the desired image path
take_photo(filename=leaf_image_path)
print('Captured leaf image saved to {}'.format(leaf_image_path))

# Function to preprocess the captured image
def preprocess_image(img_path, target_size=(224, 224)):
    try:
        img = image.load_img(img_path, target_size=target_size)
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img /= 255.0  # Normalize the image
        return img
    except Exception as e:
        print("Error loading the image:", str(e))
        return None  # Return None if there's an error

# Function to predict the leaf class from the captured image
def predict_leaf_class(img_path):
    preprocessed_img = preprocess_image(img_path)
    if preprocessed_img is not None:
        prediction = model.predict(preprocessed_img)
        predicted_class = np.argmax(prediction)
        return predicted_class
    else:
        return None

# Map the predicted class index to class labels
class_labels = {0: 'Asthma Plant', 1: 'Avaram Plant', 2: 'Balloon vine Plant', 3: 'Bellyache bush (Green) Plant', 4: 'Benghal dayflower Plant', 5: 'Big Caltrops Plant', 6: 'Black-Honey Shrub Plant', 7: 'Bristly Wild Grape Plant', 8: 'Butterfly Pea Plant', 9: 'Cape Gooseberry Plant', 10: 'Common Wireweed Plant', 11: 'Country Mallow Plant', 12: 'Crown flower Plant', 13: 'Green Chireta Plant', 14: 'Heart-leaved moonseed Plant', 15: 'Holy Basil Plant', 16: 'Indian CopperLeaf Plant', 17: 'Indian Jujube Plant', 18: 'Indian Sarsaparilla Plant', 19: 'Indian Stinging Nettle Plant', 20: 'Indian Thornapple Plant', 21: 'Indian wormwood Plant', 22: 'Ivy Gourd Plant', 23: 'Kokilaksha Plant', 24: 'Land Caltrops (Bindii) Plant', 25: 'Madagascar Periwinkle Plant', 26: 'Madras Pea Pumpkin Plant', 27: 'Malabar Catmint Plant', 28: 'Mexican Mint Plant', 29: 'Mexican Prickly Poppy Plant', 30: 'Mountain Knotgrass Plant', 31: 'Nalta Jute Plant', 32: 'Night blooming Cereus Plant', 33: 'Panicled Foldwing Plant', 34: 'Prickly Chaff Flower Plant', 35: 'Punarnava Plant', 36: 'Purple Fruited Pea Eggplant Plant', 37: 'Purple Tephrosia Plant', 38: 'Rosary Pea Plant', 39: 'Shaggy button weed Plant', 40: 'Small Water Clover Plant', 41: 'Spiderwisp Plant', 42: 'Square Stalked Vine Plant', 43: 'Stinking Passionflower Plant', 44: 'Sweet Basil Plant', 45: 'Sweet flag Plant', 46: 'Tinnevelly Senna Plant', 47: 'Trellis Vine Plant', 48: 'Velvet bean Plant', 49: 'coatbuttons Plant'}

# Capture an image using the laptop's webcam
#capture_image()

# Predict the leaf class from the captured image
captured_img_path = 'leaf_image.jpg'
predicted_class = predict_leaf_class(captured_img_path)

# Check if prediction was successful
if predicted_class is not None:
    # Get the predicted class label
    predicted_label = class_labels.get(predicted_class, 'Unknown Class')
    print('Predicted Leaf Class:', predicted_label)
else:
    print('Failed to predict leaf class. Please try again.')

# Display the captured leaf image with the predicted class label
img = PILImage.open(leaf_image_path)
plt.imshow(img)
plt.axis('off')
plt.title(f'Predicted Class: {predicted_label}')
plt.show()


import requests

def get_google_info(query, cx):
    api_key = 'YOUR_GOOGLE_API_KEY'
    search_url = f'https://www.googleapis.com/customsearch/v1?q={query}&key={api_key}&cx={cx}'

    try:
        response = requests.get(search_url)
        if response.status_code == 200:
            data = response.json()
            items = data.get('items', [])

            if items:
                # Extract relevant information from the API response
                description = items[0].get('snippet', '')
                link = items[0].get('link', '')

                print('Description of', {query},':', description)
                print('Link to the page:', link)

                # Perform a new search for habitat information
                habitat_query = f'where does {query} grow'
                habitat_search_url = f'https://www.googleapis.com/customsearch/v1?q={habitat_query}&key={api_key}&cx={cx}'
                habitat_response = requests.get(habitat_search_url)

                if habitat_response.status_code == 200:
                    habitat_data = habitat_response.json()
                    habitat_items = habitat_data.get('items', [])

                    if habitat_items:
                        habitat_description = habitat_items[0].get('snippet', '')
                        habitat_link = habitat_items[0].get('link', '')
                        print('Habitat Description:', habitat_description)
                        print('Habitat Link:', habitat_link)
                    else:
                        print('No habitat information found.')

            else:
                print('No results found.')

        else:
            print('Failed to retrieve information from Google.')

    except requests.exceptions.RequestException as e:
        print('Error:', str(e))

# Example usage:
leaf_name = predicted_label  # Update with the detected leaf name
custom_search_engine_id = 'YOUR_CUSTOM_SEARCH_ENGINE_ID'
get_google_info(leaf_name, custom_search_engine_id)


import requests

def get_google_info(query, cx):
    api_key = 'YOUR_GOOGLE_API_KEY'
    search_url = f'https://www.googleapis.com/customsearch/v1?q={query}&key={api_key}&cx={cx}'

    try:
        response = requests.get(search_url)
        if response.status_code == 200:
            data = response.json()
            items = data.get('items', [])

            if items:
                # Extract relevant information from the API response
                description = items[0].get('snippet', '')
                link = items[0].get('link', '')

                print('Description of', {query},':', description)
                print('Link to the page:', link)

                # Perform a new search for habitat information
                habitat_query = f'where does {query} grow'
                habitat_search_url = f'https://www.googleapis.com/customsearch/v1?q={habitat_query}&key={api_key}&cx={cx}'
                habitat_response = requests.get(habitat_search_url)

                if habitat_response.status_code == 200:
                    habitat_data = habitat_response.json()
                    habitat_items = habitat_data.get('items', [])

                    if habitat_items:
                        habitat_description = habitat_items[0].get('snippet', '')
                        habitat_link = habitat_items[0].get('link', '')
                        print('Habitat Description:', habitat_description)
                        print('Habitat Link:', habitat_link)
                    else:
                        print('No habitat information found.')

            else:
                print('No results found.')

        else:
            print('Failed to retrieve information from Google.')

    except requests.exceptions.RequestException as e:
        print('Error:', str(e))

# Example usage:
leaf_name = predicted_label  # Update with the detected leaf name
custom_search_engine_id = 'YOUR_CUSTOM_SEARCH_ENGINE_ID'
get_google_info(leaf_name, custom_search_engine_id)


import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Directory where your dataset is stored
data_dir = r'C:\Users\chinm\Downloads\Indian Medicinal Leaves Image Datasets\Medicinal Leaf dataset'

# Define the image dimensions and batch size
img_height, img_width = 224, 224
batch_size = 32

# Load the pre-trained VGG16 model (without the top layers)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Freeze the pre-trained layers so they are not trainable
for layer in base_model.layers:
    layer.trainable = False

# Define the number of classes in your dataset
num_classes = 80  # Updated to match local dataset

# Build the model
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')  # Use the actual number of classes
])

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Use ImageDataGenerator for data augmentation and preprocessing
train_data_gen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Splitting the data into training and validation sets
)

train_generator = train_data_gen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_data_gen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Define early stopping criteria
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model with early stopping
epochs = 40
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[early_stopping]  # Add the early stopping callback here
)

# Plot training and validation accuracy
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# Plot training accuracy and validation accuracy over epochs
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Save the model
model.save('medicinal_plant_model_VGG16_new.keras')

# pip install requests

import requests

def get_google_info(query, cx):
    api_key = 'YOUR_GOOGLE_API_KEY'
    search_url = f'https://www.googleapis.com/customsearch/v1?q={query}&key={api_key}&cx={cx}'

    try:
        response = requests.get(search_url)
        if response.status_code == 200:
            data = response.json()
            items = data.get('items', [])

            if items:
                # Extract relevant information from the API response
                description = items[0].get('snippet', '')
                link = items[0].get('link', '')

                print('Description:', description)
                print('Link to the page:', link)

                # Perform a new search for habitat information
                habitat_query = f'{query} habitat'
                habitat_search_url = f'https://www.googleapis.com/customsearch/v1?q={habitat_query}&key={api_key}&cx={cx}'
                habitat_response = requests.get(habitat_search_url)

                if habitat_response.status_code == 200:
                    habitat_data = habitat_response.json()
                    habitat_items = habitat_data.get('items', [])

                    if habitat_items:
                        habitat_description = habitat_items[0].get('snippet', '')
                        habitat_link = habitat_items[0].get('link', '')
                        print('Habitat Description:', habitat_description)
                        print('Habitat Link:', habitat_link)
                    else:
                        print('No habitat information found.')

            else:
                print('No results found.')

        else:
            print('Failed to retrieve information from Google.')

    except requests.exceptions.RequestException as e:
        print('Error:', str(e))

# Example usage:
leaf_name = 'Tulsi Plant'   # Update with the detected leaf name
custom_search_engine_id = 'YOUR_CUSTOM_SEARCH_ENGINE_ID'
get_google_info(leaf_name, custom_search_engine_id)


import requests

def get_google_info(query, cx):
    api_key = 'YOUR_GOOGLE_API_KEY'
    search_url = f'https://www.googleapis.com/customsearch/v1?q={query}&key={api_key}&cx={cx}'

    try:
        response = requests.get(search_url)
        if response.status_code == 200:
            data = response.json()
            items = data.get('items', [])

            if items:
                # Extract relevant information from the API response
                description = items[0].get('snippet', '')
                print('Description:', description)

                # You can parse the description to extract more specific information if needed

            else:
                print('No results found.')

        else:
            print('Failed to retrieve information from Google.')

    except requests.exceptions.RequestException as e:
        print('Error:', str(e))

# Example usage:
leaf_name = 'Trellis vine'  #predicted_label   Update with the detected leaf name
custom_search_engine_id = 'YOUR_CUSTOM_SEARCH_ENGINE_ID'
get_google_info(leaf_name, custom_search_engine_id)


import requests

def get_google_info(query, cx):
    api_key = 'YOUR_GOOGLE_API_KEY'
    search_url = f'https://www.googleapis.com/customsearch/v1?q={query}&key={api_key}&cx={cx}'

    try:
        response = requests.get(search_url)
        if response.status_code == 200:
            data = response.json()
            items = data.get('items', [])

            if items:
                # Extract relevant information from the API response
                description = items[0].get('snippet', '')
                link = items[0].get('link', '')

                print('Description:', description)
                print('Link to the page:', link)

                # You can parse the description to extract more specific information if needed

            else:
                print('No results found.')

        else:
            print('Failed to retrieve information from Google.')

    except requests.exceptions.RequestException as e:
        print('Error:', str(e))

# Example usage:
leaf_name = 'Sweet Flag Plant'  # Update with the detected leaf name
custom_search_engine_id = 'YOUR_CUSTOM_SEARCH_ENGINE_ID'
get_google_info(leaf_name, custom_search_engine_id)


import requests

def get_google_info(query, cx):
    api_key = 'YOUR_GOOGLE_API_KEY'
    search_url = f'https://www.googleapis.com/customsearch/v1?q={query}&key={api_key}&cx={cx}'

    try:
        response = requests.get(search_url)
        if response.status_code == 200:
            data = response.json()
            items = data.get('items', [])

            if items:
                # Extract relevant information from the API response
                description = items[0].get('snippet', '')
                link = items[0].get('link', '')

                print('Description:', description)
                print('Link to the page:', link)

                # Perform a new search for habitat information
                habitat_query = f'where does {query} grow'
                habitat_search_url = f'https://www.googleapis.com/customsearch/v1?q={habitat_query}&key={api_key}&cx={cx}'
                habitat_response = requests.get(habitat_search_url)

                if habitat_response.status_code == 200:
                    habitat_data = habitat_response.json()
                    habitat_items = habitat_data.get('items', [])

                    if habitat_items:
                        habitat_description = habitat_items[0].get('snippet', '')
                        habitat_link = habitat_items[0].get('link', '')
                        print('Habitat Description:', habitat_description)
                        print('Habitat Link:', habitat_link)
                    else:
                        print('No habitat information found.')

            else:
                print('No results found.')

        else:
            print('Failed to retrieve information from Google.')

    except requests.exceptions.RequestException as e:
        print('Error:', str(e))

# Example usage:
leaf_name = 'Tulsi Plant'  # Update with the detected leaf name
custom_search_engine_id = 'YOUR_CUSTOM_SEARCH_ENGINE_ID'
get_google_info(leaf_name, custom_search_engine_id)


import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Directory where your dataset is stored
data_dir = r'C:\Users\chinm\Downloads\Indian Medicinal Leaves Image Datasets\Medicinal Leaf dataset'

# Define the image dimensions and batch size
img_height, img_width = 224, 224
batch_size = 32

# Load the pre-trained InceptionV3 model (without the top layers)
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Freeze the pre-trained layers so they are not trainable
for layer in base_model.layers:
    layer.trainable = False

# Define the number of classes in your dataset
num_classes = 80  # Updated to match local dataset

# Build the model
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')  # Use the actual number of classes
])

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Use ImageDataGenerator for data augmentation and preprocessing
train_data_gen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Splitting the data into training and validation sets
)

train_generator = train_data_gen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_data_gen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Define early stopping criteria
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model with early stopping
epochs = 40
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[early_stopping]  # Add the early stopping callback here
)

# Plot training accuracy and validation accuracy over epochs
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()


# Save the model
model.save('medicinal_plant_model_INCEPTION.keras')

from google.colab.patches import cv2_imshow
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image as PILImage
import matplotlib.pyplot as plt
from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode
import tensorflow as tf


# Load the trained model
model = load_model('/content/medicinal_plant_model_INCEPTION.keras')

# Function to capture a photo using the webcam
def take_photo(filename='leaf_image.jpg', quality=0.8):
    js = Javascript('''
        async function takePhoto(quality) {
            const div = document.createElement('div');
            const capture = document.createElement('button');
            capture.textContent = 'Capture';
            div.appendChild(capture);

            const video = document.createElement('video');
            video.style.display = 'block';
            const stream = await navigator.mediaDevices.getUserMedia({ 'video': true });

            document.body.appendChild(div);
            div.appendChild(video);
            video.srcObject = stream;
            await video.play();

            // Resize the output to fit the video element.
            google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

            // Wait for Capture to be clicked.
            await new Promise((resolve) => capture.onclick = resolve);

            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);
            stream.getVideoTracks()[0].stop();
            div.remove();
            return canvas.toDataURL('image/jpeg', quality);
        }
    ''')
    display(js)

    data = eval_js('takePhoto({})'.format(quality))
    binary = b64decode(data.split(',')[1])
    with open(filename, 'wb') as f:
        f.write(binary)
    return filename

# Capture a leaf image
leaf_image_path = 'leaf_image.jpg'  # Update with the desired image path
take_photo(filename=leaf_image_path)
print('Captured leaf image saved to {}'.format(leaf_image_path))

# Function to preprocess the captured image
def preprocess_image(img_path, target_size=(224, 224)):
    try:
        img = image.load_img(img_path, target_size=target_size)
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img /= 255.0  # Normalize the image
        return img
    except Exception as e:
        print("Error loading the image:", str(e))
        return None  # Return None if there's an error

# Function to predict the leaf class from the captured image
def predict_leaf_class(img_path):
    preprocessed_img = preprocess_image(img_path)
    if preprocessed_img is not None:
        prediction = model.predict(preprocessed_img)
        predicted_class = np.argmax(prediction)
        return predicted_class
    else:
        return None

# Map the predicted class index to class labels
class_labels = {0: 'Asthma Plant.zip', 1: 'Avaram.zip', 2: 'Balloon vine.zip', 3: 'Bellyache bush (Green).zip', 4: 'Benghal dayflower.zip', 5: 'Big Caltrops.zip', 6: 'Black-Honey Shrub.zip', 7: 'Bristly Wild Grape.zip', 8: 'Butterfly Pea.zip', 9: 'Cape Gooseberry.zip', 10: 'Common Wireweed.zip', 11: 'Country Mallow.zip', 12: 'Crown flower.zip', 13: 'Green Chireta.zip', 14: 'Holy Basil.zip', 15: 'Indian CopperLeaf.zip', 16: 'Indian Jujube.zip', 17: 'Indian Sarsaparilla.zip', 18: 'Indian Stinging Nettle.zip', 19: 'Indian Thornapple.zip', 20: 'Indian wormwood.zip', 21: 'Ivy Gourd.zip', 22: 'Kokilaksha.zip', 23: 'Land Caltrops (Bindii).zip', 24: 'Madagascar Periwinkle.zip', 25: 'Madras Pea Pumpkin.zip', 26: 'Malabar Catmint.zip', 27: 'Mexican Mint.zip', 28: 'Mexican Prickly Poppy.zip', 29: 'Mountain Knotgrass.zip', 30: 'Nalta Jute.zip', 31: 'Night blooming Cereus.zip', 32: 'Panicled Foldwing.zip', 33: 'Prickly Chaff Flower.zip', 34: 'Punarnava.zip', 35: 'Purple Fruited Pea Eggplant.zip', 36: 'Purple Tephrosia.zip', 37: 'Rosary Pea.zip', 38: 'Shaggy button weed.zip', 39: 'Small Water Clover.zip', 40: 'Spiderwisp.zip', 41: 'Square Stalked Vine.zip', 42: 'Stinking Passionflower.zip', 43: 'Sweet Basil.zip', 44: 'Sweet flag.zip', 45: 'Tinnevelly Senna.zip', 46: 'Trellis Vine.zip', 47: 'Velvet bean.zip', 48: 'coatbuttons.zip', 49: 'heart-leaved moonseed.zip'}  # Update with your actual class labels


# Capture an image using the laptop's webcam
#capture_image()

# Predict the leaf class from the captured image
captured_img_path = 'leaf_image.jpg'
predicted_class = predict_leaf_class(captured_img_path)

# Check if prediction was successful
if predicted_class is not None:
    # Get the predicted class label
    predicted_label = class_labels.get(predicted_class, 'Unknown Class')
    print('Predicted Leaf Class:', predicted_label)
else:
    print('Failed to predict leaf class. Please try again.')

# Display the captured leaf image with the predicted class label
img = PILImage.open(leaf_image_path)
plt.imshow(img)
plt.axis('off')
plt.title(f'Predicted Class: {predicted_label}')
plt.show()


# Save the model for use in app.py
model.save('leaf_model.keras')
print('Model saved as leaf_model.keras')
