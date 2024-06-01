from typing import Union
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import pipeline
from pydantic import BaseModel
from PIL import Image
import io
import numpy as np
from deepface import DeepFace

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Cargar el modelo de analisis de sentimientos
clasificador = pipeline(
    "sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment"
)

# Solo recibe 250 car


@app.get("/analizar")
def analizar_sentimiento_hf(texto: str):
    resultado = clasificador(texto)
    return {
        "texto": texto,
        "label": resultado[0]["label"],
        "score": resultado[0]["score"],
    }


class ImagePredictionResponse(BaseModel):
    prediction: str

@app.post("/predecir-imagen", response_model=ImagePredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    open_cv_image = np.array(image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()

    # Realizar la predicción de emociones
    result = DeepFace.analyze(open_cv_image, actions=['emotion'])

    # Obtener la emoción principal
    emotion = result["dominant_emotion"]
    return ImagePredictionResponse(prediction=emotion)
