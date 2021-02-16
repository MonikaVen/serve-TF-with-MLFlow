import mlflow
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import cv2
import numpy as np
test_path = "./data/test.parquet"

def get_photos(uuids, preprocessing_fn, photos_path):
    images = []
    for uuid in tqdm(uuids, total=len(uuids)):
        img = preprocessing_fn(cv2.resize(cv2.cvtColor(cv2.imread(f"{photos_path}/{uuid}.jpeg"), cv2.COLOR_BGR2RGB), (299, 299)))
        images.append(img)
    return images



test = pd.read_parquet(test_path)
preprocessing_fn = tf.keras.applications.xception.preprocess_input
test_imgs = get_photos(test["uuid"].values.tolist(), preprocessing_fn, photos_path="photos")
data = np.array(test_imgs)

logged_model = 'file:///home/monika/Documents/Test/mlruns/0/084d67848b6543379c036509f78aecaf/artifacts/model'
# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

pred = loaded_model.predict(pd.DataFrame({"inference": data}))
print(pred)

# curl http://127.0.0.1:5000/invocations -H 'Content-Type: application/json' -d '{
#     "columns": ["a", "b", "c"],
#     "data": [[1, 2, 3], [4, 5, 6]]
# }'