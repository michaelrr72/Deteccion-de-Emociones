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
import requests
from io import BytesIO
import cv2
import numpy as np

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Cargar el modelo
with tf.keras.utils.custom_object_scope({"KerasLayer": hub.KerasLayer}):
    modelo_cargado = load_model("mm/modelo_entrenado.h5")


class ImagePredictionResponse(BaseModel):
    prediction: str


@app.post("/predecir-imagen", response_model=ImagePredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    respuesta = requests.get(contents)
    img = Image.open(BytesIO(respuesta.content))
    img = np.array(img).astype(float) / 255

    img = cv2.resize(img, (224, 224))
    prediction = modelo_cargado.predict(img.reshape(-1, 224, 224, 3))
    emotion = np.argmax(prediction)

    # Emoción correspondiente
    emotion_labels = [
        "disgustado",
        "enojado",
        "feliz",
        "neutral",
        "sorprendido",
        "temeroso",
        "triste",
    ]
    predicted_emotion = emotion_labels[emotion]

    return ImagePredictionResponse(prediction=predicted_emotion)
