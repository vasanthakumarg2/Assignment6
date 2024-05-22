from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
from tensorflow import keras
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import argparse
import uvicorn
from keras.layers import InputLayer
from scipy import ndimage

app = FastAPI()


## Parse path to the model from command line
def parse_command():

    parser = argparse.ArgumentParser(description='Load model')
    parser.add_argument('path',type=str, help="Path of the model")
    
    ## parse the arguements
    args = parser.parse_args()

    return args.path

## Load the model
def load(file_path):

    model = load_model(file_path)
    return model

## Format image to the required size
def format_image(image):

    ##convert to grayscale
    img_gray = image.convert('L') 
    ##resize and normalzie
    img_resized = np.array(img_gray.resize((28, 28)))/255.0

    ##center the image
    cy, cx = ndimage.center_of_mass(img_resized)
    rows, cols = img_resized.shape
    shiftx = np.round(cols / 2.0 - cx).astype(int)
    shifty = np.round(rows / 2.0 - cy).astype(int)
    
    ##translate the image
    M = np.float32([[1, 0, shiftx], [0, 1, shifty]])
    img_centered = ndimage.shift(img_resized, (shifty, shiftx), cval=0)
    
    ##flatten the image
    return img_centered.flatten()

## Predict digit from the input image
def predict_digit(model, img):

    prediction = model.predict(img.reshape(1,-1))

    ##select the max
    return str(max(enumerate(prediction), key=lambda x: x[1])[0])

## API endpoint : /mnist that will read the input image and convert it into serialized array through an asynchronous function
@app.post('/predict')
async def predict(file: UploadFile = File(...)):

    ##wait till the file is read
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))

    ##convert to grayscale and normalize
    img_array = format_image(img)

    ##take the path to model through command line and load model
    path = parse_command()
    model = load(path)
    
    ##predict the digit
    pred = predict_digit(model, img_array)

    return {"digit": pred}

if __name__ == '__main__':
    uvicorn.run("task2:app", reload=True)