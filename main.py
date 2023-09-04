import os
import io
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from pyngrok import ngrok
from PIL import Image
from rembg import remove
import cv2
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
import rembg  
import time

app = FastAPI()

model_path = "model_1.h5"
model_path_2 = "model_2.h5"
model_1 = tf.keras.models.load_model(model_path)
model_2 = tf.keras.models.load_model(model_path_2)

class_names = ['Miner', 'Nodisease', 'Phoma', 'Rust']

def predict_image(image_data, model_type):

    if model_type == "model_1":
      model = model_1
    elif model_type == "model_2":
      model = model_2
    
    image = Image.open(io.BytesIO(image_data))
    temp = np.array(image)

    temp = remove(temp)

    if temp.shape[-1] == 4:
        temp = temp[:, :, :3]

    temp = cv2.resize(temp, (150, 150))
    result_image = Image.fromarray(temp)

    start_time = time.time()
    prediction = model.predict(np.expand_dims(result_image, axis=0))
    end_time = time.time()

    percentages = [round(p * 100, 2) for p in prediction[0]]

    result = {class_names[i]: percentages[i] for i in range(len(class_names))}
    result['hasil'] = class_names[percentages.index(max(percentages))]
    result['lama_prediksi'] = f"{(end_time - start_time):.5f} seconds"

    return result

@app.post("/predict/")
async def predict(file: UploadFile, model_type: str = Form(...)):
    image_data = await file.read()
    prediction_result = predict_image(image_data, model_type)

    return {"prediction": prediction_result}
