# Emotion Detection API

Este proyecto proporciona una API basada en FastAPI para la detección de emociones en imágenes utilizando un modelo de TensorFlow entrenado. También incluye una interfaz web simple para subir imágenes y obtener predicciones de emociones.

## Requisitos

- Python 3.6 o superior
- pip (gestor de paquetes de Python)

## Instalación

1. Clona este repositorio en tu máquina local:

    ```bash
    git clone https://github.com/michaelrr72/emotion-detection-api.git
    cd emotion-detection-api
    ```

2. Crea y activa un entorno virtual:

    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows usa `venv\Scripts\activate`
    ```

3. Instala las dependencias necesarias:

    ```bash
    pip install fastapi uvicorn transformers tensorflow tensorflow-hub pillow numpy
    ```

## Modelo Preentrenado

Asegúrate de tener el modelo preentrenado en el directorio `mm` con el nombre `modelo_entrenado.h5`.

## Ejecución del Servidor

1. Ejecuta el servidor FastAPI:

    ```bash
    uvicorn main:app --port 3001 --reload
    ```

2. Abre tu navegador y ve a `http://localhost:3001` para acceder a la interfaz web.