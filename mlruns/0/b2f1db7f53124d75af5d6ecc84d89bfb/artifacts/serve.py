import mlflow
import tensorflow as tf
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec
import tempfile
import os

import psycopg2
import sys
MODEL_NAME="clothes"
EVALUATION_METRIC="accuracy"
TRAIN_STEPS=1000
BATCH_SIZE=100
ACCURACY=0.7
INPUT_EXAMPLE="{image: image}"
os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:8000/'
# con = psycopg2.connect(database='mlflow_db', user='mlflow_user',
#     password='mlflow')
# cur = con.cursor()
# cur.execute('SELECT version()')

#from the train script
dictionary = {
  "model_name": MODEL_NAME,
  "train_steps": TRAIN_STEPS,
  "batch_size": BATCH_SIZE,
  "accuracy": ACCURACY
}

input_schema = Schema([
  ColSpec("double", "sepal length (cm)"),
  ColSpec("double", "sepal width (cm)"),
  ColSpec("double", "petal length (cm)"),
  ColSpec("double", "petal width (cm)"),
])
output_schema = Schema([ColSpec("long")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)



model_path = "./models/clothes/1/"
#model_uri = "mysql+pymysql://'mlflow-user':'password'@localhost:3306/mlflowruns"
sqlite = "sqlite:///store.db"
postgres = "postgresql+psycopg2://mlflow_user:mlflow@localhost/mlflow_db"
def main():
    with mlflow.start_run(experiment_id=0, run_name='test_clothes'):
        
        mlflow.log_dict(dictionary, "data.json")
        mlflow.set_tag("remark", "Testing the existing model")
        mlflow.tensorflow.autolog()
        model = tf.saved_model.load(model_path)
        inputs = '[{"name": "sepal length (cm)", "type": "double"}, {"name": "sepal width (cm)", "type": "double"}, {"name": "petal length (cm)", "type": "double"}, {"name": "petal width (cm)", "type": "double"}]'
        outputs = '[{"type": "integer"}]'
        mlflow.log_param("train_steps", TRAIN_STEPS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        # Logging serve code
        mlflow.log_artifact(local_path = './serve.py')
        mlflow.tensorflow.log_model(tf_saved_model_dir = model_path, tf_meta_graph_tags = ['serve'], tf_signature_def_key = 'serving_default', artifact_path='model',registered_model_name=MODEL_NAME, input_example=INPUT_EXAMPLE)
        

if __name__ == "__main__":
    main()