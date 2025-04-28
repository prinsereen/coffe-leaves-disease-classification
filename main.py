import os
import io
import time
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import cv2
from rembg import remove

app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ganti ini kalau mau lebih ketat
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Models
model_1 = tf.keras.models.load_model("model_1.h5")
model_2 = tf.keras.models.load_model("model_2.h5")

# Class names
class_names = ['Miner', 'Nodisease', 'Phoma', 'Rust']

# Prediction function
def predict_image(image_data: bytes, model_type: str):
    # Pilih model
    model = model_1 if model_type == "model_1" else model_2

    # Buka gambar
    image = Image.open(io.BytesIO(image_data))
    temp = np.array(image)

    # Remove background
    temp = remove(temp)

    # Pastikan hanya 3 channel (RGB)
    if temp.shape[-1] == 4:
        temp = temp[:, :, :3]

    # Resize ke ukuran model
    temp = cv2.resize(temp, (150, 150))

    # Normalize pixel (0-1)
    temp = temp / 255.0

    # Prediksi
    temp = np.expand_dims(temp, axis=0)  # Tambah batch dimension
    start_time = time.time()
    prediction = model.predict(temp)
    end_time = time.time()

    # Convert hasil prediksi
    percentages = [float(round(p * 100, 2)) for p in prediction[0]]
    result = {class_names[i]: percentages[i] for i in range(len(class_names))}
    result['hasil'] = class_names[int(np.argmax(percentages))]
    result['lama_prediksi'] = f"{(end_time - start_time):.5f} seconds"

    return result

# API Endpoint
@app.post("/predict/")
async def predict(file: UploadFile = Form(...), model_type: str = Form(...)):
    try:
        image_data = await file.read()
        prediction_result = predict_image(image_data, model_type)
        return JSONResponse(content={"success": True, "prediction": prediction_result})
    except Exception as e:
        return JSONResponse(content={"success": False, "error": str(e)})
