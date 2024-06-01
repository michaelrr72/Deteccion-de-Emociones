import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import load_model
from typing import Union
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from PIL import Image
import io
import numpy as np

with tf.keras.utils.custom_object_scope({'KerasLayer': hub.KerasLayer}):
    modelo_cargado = load_model('mm/modelo_entrenado.h5')

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

class ImagePredictionResponse(BaseModel):
    prediction: str

def preprocess_image(image: Image.Image) -> np.ndarray:
    # Aquí puedes ajustar el tamaño de la imagen y hacer la normalización necesaria para tu modelo
    image = image.resize((224, 224))  # Ajusta según el tamaño de entrada de tu modelo
    image_array = np.array(image) / 255.0  # Normaliza los valores de píxel
    image_array = np.expand_dims(image_array, axis=0)  # Añade una dimensión para el batch
    return image_array

@app.post("/predecir-imagen", response_model=ImagePredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    processed_image = preprocess_image(image)
    
    # Realiza la predicción
    prediction = modelo_cargado.predict(processed_image)
    emotion = np.argmax(prediction)  # Esto depende de cómo tu modelo devuelve la predicción

    # Mapea el índice a la emoción correspondiente
    emotion_labels = ["feliz", "triste", "enojado", "sorprendido"]  # Ajusta según tu modelo
    predicted_emotion = emotion_labels[emotion]

    return ImagePredictionResponse(prediction=predicted_emotion)