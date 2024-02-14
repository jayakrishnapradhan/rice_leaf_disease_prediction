from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from datetime import datetime
from ricePrediction.settings import BASE_DIR
from django.http import HttpResponse
import pickle

import pandas as pd
import numpy as np 
import tensorflow as tf
import seaborn as sns 
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier

import os
from os import listdir
from PIL import Image
import keras
from keras.models import Sequential
from tensorflow.keras import layers
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow_hub as hub
from tensorflow.keras import utils
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
from keras.models import load_model


data_lines = []

for i in range(1, 5):
    with open(f"rice_disease_prediction_model_accuracy_7500_{i}.h5", "rb") as f:
        data_lines.extend(f.readlines())

with open("rice_disease_prediction_model_accuracy_7500.h5", "wb") as f:
    f.writelines(data_lines)

# Create your views here.
def predict(request):

    model = load_model(os.path.join(BASE_DIR, "rice_disease_prediction_model_accuracy_7500.h5"))

    # Load the image and preprocess it
    image_path = request.session["uploaded_image"]
    image_path = os.path.join(BASE_DIR, "uploads", image_path)

    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make predictions
    predictions = model.predict(img_array)

    # Assuming 'predictions' is the array of predicted probabilities
    predicted_class_index = np.argmax(predictions)

    # Manually map the class index to your custom class labels
    custom_class_labels = ["Bacterial Leaf Blight", "Brown Spot", "Leaf Smut"]
    predicted_class_label = custom_class_labels[predicted_class_index]
    
    accuracy = predictions[0][predicted_class_index]
    return render(request, "result.html", {"predicted_class_label": predicted_class_label, "accuracy": accuracy*100})

def home(request):
    if request.method == "POST":
        image = request.FILES['inputImage']

        # Validate file extension
        allowed_extensions = ['jpg', 'jpeg', 'png']
        file_extension = image.name.split('.')[-1].lower()
        if file_extension not in allowed_extensions:
            return HttpResponse("invalid file format")

        # Save the uploaded file with a timestamped name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_uploaded.{file_extension}"
        
        # Set the path to the "uploads" folder
        upload_folder = 'uploads'
        upload_path = os.path.join(BASE_DIR, upload_folder, filename)

        # Save the image to the "uploads" folder
        fs = FileSystemStorage()
        fs.save(upload_path, image)

        # Store the filename in a session variable
        request.session['uploaded_image'] = filename

        return redirect(predict)  # Redirect to a success page or another view
    return render(request, 'index.html')