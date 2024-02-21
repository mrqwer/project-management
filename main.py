from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import numpy as np  # For array manipulation
from PIL import Image  # For image processing
import io  # For handling bytes streams
import tensorflow as tf  # For using your model
from pydantic import BaseModel

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from all origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)
model = tf.keras.models.load_model('modelW1.h5')


classes = ['Impressionism', 'Realism', 'Romanticism', 'Expressionism',
 'Post-Impressionism', 'Baroque', 'Art Nouveau (Modern)', 'Unknown',
 'Surrealism', 'Symbolism', 'Neoclassicism', 'Abstract Expressionism',
 'Rococo', 'Northern Renaissance', 'Cubism', 'Pop Art', 'Academicism',
 'Minimalism', 'Conceptual Art', 'Art Informel', 'Naïve Art (Primitivism)',
 'Early Renaissance']

class Style(BaseModel):
    name: str
    confidence: str

class ImageResponse(BaseModel):
    name: str
    predictions: list[Style]


def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize((224, 224))  # Resize the image
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def predict(image):
    prediction = model.predict(image)
    return prediction[0]

@app.post("/predict/")
async def predict_images(files: List[UploadFile]) -> list[ImageResponse]:
    predictions = []
    for file in files:
        contents = await file.read()
        preprocessed_image = preprocess_image(contents)
        prediction = predict(preprocessed_image)
        prediction_styles = []
        for i in range(len(prediction)):
            class_name = classes[i]
            confidence = prediction[i] * 100
            style = Style(name=class_name, confidence="{:.2%}".format(confidence))
            prediction_styles.append(style)

        prediction_response = ImageResponse(name=file.filename, predictions=prediction_styles)
        predictions.append(prediction_response)

    return predictions

