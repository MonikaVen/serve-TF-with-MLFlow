import numpy as np
import requests
import tensorflow as tf
import json
import pandas
import cv2
import mlflow
SIZE = 128
MODEL_URI = "http://127.0.0.1:5000/invocations"

image_path = '/home/monika/Downloads/v2.7_2030551104/v2.7/photos/00_04e2a_JdxtpYpFmHNLrieuPSBqEqnM.jpeg'

sample_input = {
    "columns": [
        "alcohol",
        "chlorides",
        "citric acid",
        "density",
        "fixed acidity",
        "free sulfur dioxide",
        "pH",
        "residual sugar",
        "sulphates",
        "total sulfur dioxide",
        "volatile acidity"
    ],
    "data": [
        [8.8, 0.045, 0.36, 1.001, 7, 45, 3, 20.7, 0.45, 170, 0.27]
    ]
}
# response = requests.post(
#               url=webservice.scoring_uri, data=json.dumps(sample_input),
#               headers={"Content-type": "application/json"})
# response_json = json.loads(response.text)
# print(response_json)

image_path = '/home/monika/Downloads/v2.7_2030551104/v2.7/photos/00_04e2a_JdxtpYpFmHNLrieuPSBqEqnM.jpeg'

model_path = "./models/clothes/1/"

import pandas as pd
import base64
def get_prediction(image_path):
    host = 'http://0.0.0.0:'
    port = '8000'
    filenames = [image_path]
    def read_image(x):
        with open(x, "rb") as f:
            return f.read()    
    SIZE=128
    # image = tf.keras.preprocessing.image.load_img(image_path, target_size=(SIZE, SIZE))
    # image = tf.keras.preprocessing.image.img_to_array(image)
    # image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    # image = np.expand_dims(image, axis=0)
    # image_data = image
    # data = pd.DataFrame(data=image_data.tolist()).to_json()


    # print(image_data.ndim)
    # PRETRAINED_MODEL_INPUT_SHAPE = (299, 299, 3)

    # data = pd.DataFrame(
    #     data=image_data, columns=["image"]
    # ).to_json(orient="split")
    # print(data)
    # response = requests.post(
    #     url=MODEL_URI,
    #     data=data,
    #     headers={"Content-Type": "application/json; format=pandas-split"},
    # )
    # print(pd.DataFrame(data=image_data.tolist()).shape)
    # data = pd.DataFrame(
    #     data=image_data
    # ).to_json(orient="split")
    # print(data)
    # print(response.text)


    # image = tf.keras.preprocessing.image.load_img(image_path, target_size=(SIZE, SIZE))
    # image = tf.keras.preprocessing.image.img_to_array(image)
    # image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    # image = np.expand_dims(image, axis=0)
    images = []

    preprocessing_fn = tf.keras.applications.xception.preprocess_input
    img = preprocessing_fn(cv2.resize(cv2.cvtColor(cv2.imread(f"{image_path}"), cv2.COLOR_BGR2RGB), (299, 299)))
    images.append(img)
    image = pandas.DataFrame(images)
    print(image.ndim)

    # data = json.dumps({'instances': image.tolist() })
    # image_list = image.tolist()
    # pandas.DataFrame.to_json(image_list, orient='split')
    # response = requests.post(MODEL_URI, data=data.encode(), headers={"Content-type": "application/json"})
    # print(response.text)
    # result = json.loads(response.text)

get_prediction(image_path)