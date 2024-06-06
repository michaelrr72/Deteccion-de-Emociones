from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import requests
from io import BytesIO
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tf_keras as tfk
from tensorflow.keras.models import load_model

# Cargar el modelo
with tf.keras.utils.custom_object_scope({'KerasLayer': hub.KerasLayer}):
    #modelo_cargado = load_model('mm/modelo_entrenado.keras')
    modelo_cargado = load_model('mm/modelo_entrenado.keras')

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


def categorizar(url):
  respuesta = requests.get(url)
  img = Image.open(BytesIO(respuesta.content))
  img = np.array(img).astype(float)/255

  img = cv2.resize(img, (224,224))
  prediccion = modelo_cargado.predict(img.reshape(-1, 224, 224, 3))
  return np.argmax(prediccion[0], axis=-1), prediccion

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
