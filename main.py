from typing import Union
from fastapi import FastAPI
from transformers import pipeline


app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

# Cargar el modelo de analisis de sentimientos
clasificador = pipeline(
    "sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment"
)

#Solo recibe 250 car

@app.get("/analizar")
def analizar_sentimiento_hf(texto: str):
    resultado = clasificador(texto)
    return {
        "texto": texto,
        "label": resultado[0]["label"],
        "score": resultado[0]["score"],
    }
