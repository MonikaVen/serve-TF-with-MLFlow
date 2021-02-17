import os
import string
import random
import json
import requests
import time
import numpy as np
import tensorflow as tf
import logging 
from flask import Flask, request, redirect, url_for, render_template
from flask_bootstrap import Bootstrap
import datetime
from PIL import Image
app = Flask(__name__)
Bootstrap(app)

"""
Constants
"""
MODEL_URI = 'http://tensorflow-serving:8501/v1/models/clothes:predict'
OUTPUT_DIR = 'static'
CLASSES_TYPE = ['DRESSES', 'TROUSERS', 'SHORTS_CAPRIS', 'SKIRTS', 'PULLOVERS_SWEATERS']
SIZE = 5
log_path = os.path.join('logs', 'app.log')

logging.basicConfig(filename=log_path, level=logging.DEBUG)
"""
Utility functions
"""
def log_debug(message):
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    logging.debug(f'{st} {message}')
    return 1

def log_error(message):
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    logging.error(f'{st} {message}')
    return 1

def generate_filename():
    return ''.join(random.choices(string.ascii_lowercase, k=20)) + '.jpg'

def get_prediction(image_path):
    try:
        log_debug("Starting to predict...")
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(SIZE, SIZE))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        image = np.expand_dims(image, axis=0)
        data = json.dumps({'instances': image.tolist() })
        response = requests.post(MODEL_URI, data=data.encode())
        result = json.loads(response.text)
        prediction = result['predictions'][0]
        category_prediction = prediction['category_output']
        log_debug(category_prediction)
        sparkling_prediction = int(prediction['sparkling_output'][0] > 0.5)
        striped_prediction = int(prediction['striped_output'][0] > 0.5)
        floral_prediction = int(prediction['floral_output'][0] > 0.5)
        index_max = np.argmax(category_prediction)
        sparkling = 'YES' if sparkling_prediction else 'NO'
        striped = 'YES' if striped_prediction else 'NO'
        floral = 'YES' if floral_prediction else 'NO'
        class_name = CLASSES_TYPE[index_max] + ', SPARKLING: ' + sparkling + ', STRIPED: ' + striped + ', FLORAL: ' + floral
        original_prediction = result
        log_debug("Successfully predicted.")
        log_debug(class_name)
        return class_name, original_prediction
    except:
        log_error("There were errors while running a prediction.")

# Routes

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            log_debug("Uploading image...")
            start = time.perf_counter()
            #DATA VALIDATION
            uploaded_file = request.files['file']
            if(uploaded_file.content_type[:5] != 'image'):
                log_error("Wrong file format. Please upload an image.")
                return {'result': 'Wrong file type', 'filetype': uploaded_file.content_type}
            if uploaded_file.filename != '':
                if uploaded_file.filename[-4:] in ['.jpg', '.png', 'jpeg']:
                    log_debug("Good image file format. Processing ...")
                    image_path = os.path.join(OUTPUT_DIR, generate_filename())
                    uploaded_file.save(image_path)
                    class_name, original_prediction = get_prediction(image_path)
                    request_time = time.perf_counter() - start
                    result = {
                        'class_name': class_name,
                        'path_to_image': image_path,
                        'size': SIZE,
                        'time': f'{round(request_time*1000, 2)} ms',
                    }
                    log_debug('Done predicting.')
                    log_debug(f'Prediction time: {round(request_time*1000, 2)} ms')
                    return result
        except:
            log_error("Failed to upload an image")
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5001')