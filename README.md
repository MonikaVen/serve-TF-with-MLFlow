# MLOps: From Training to Serving

## Dependencies

Make sure you have python, tensorflow and postgres installed:

`sudo apt install python3 python3-pip libgl1-mesa-glx postgresql postgresql-contrib postgresql-server-dev-all gcc`

MLFlow needs conda. For our purposes miniconda will do:

https://docs.conda.io/en/latest/miniconda.html


## Install MLFlow:

`pip3 install mlflow==1.13.1`

Create a backend store for MLFlow:

`sudo -u postgres psql`

`CREATE DATABASE mlflow_db;
CREATE USER mlflow_user WITH ENCRYPTED PASSWORD 'mlflow';
GRANT ALL PRIVILEGES ON DATABASE mlflow_db TO mlflow_user;`
`\q`

`pip3 install psycopg2-binary`

Add photos inside a 'Vinted_Serve_MLFlow/photos' folder.
You can find the data photos at:

https://drive.google.com/drive/folders/1Cj8QGqjl-5q2DroQA50mrMEiceEri2xn?usp=sharing


## Training phase and logging model metrics:

Make sure you are tracking the model:
`export MLFLOW_TRACKING_URI='http://localhost:8000'`

Run MLFlow tracking server with storage:

`mlflow server --backend-store-uri postgresql://mlflow_user:mlflow@localhost/mlflow_db --default-artifact-root ./mlruns -h 0.0.0.0 -p 8000`

Train, log data and register the model with:

`mlflow run .`

Wait until conda downloads all the dependencies.
Then the code will be run with parameters that are defined in MLProject.

(`python3 main.py` would also work with all the dependencies).

(Another option to access UI without a model store is `mlflow ui`, it runs on :5000, so
make sure you update environment variable if you use this one
`export MLFLOW_TRACKING_URI='http://localhost:8000`).

Click on a model in a table and you will see all the parameters logged and information to run the model.

## Serving: MLFlow

You can serve the model with it's artifact:

`mlflow models serve -m "./mlruns/0/b1a6b02eb7cb4537874dff1d678b2ac9/artifacts/model" -h 0.0.0.0`

BUG: MLFlow does not support data with more than  2Dimensions. 
WORKAROUND:
Here I tried to convert everything into base64,
however, ran out of time trying to make an adapter for a model.
Served in this way model would provide production logs.

https://github.com/mlflow/mlflow/issues/2830

Logs can be found at mllogs folder.
Parameters are logged and also displayed in MLFlow.

For the display of metrics and parameters also check MLFlow ui at:

http://localhost:8000


You will see something like this:

Parameters
Name	Value
batch_size	8
data_limit	40
dp_rate	0.3
epochs	3
opt_amsgrad	False
opt_beta_1	0.9
opt_beta_2	0.999
opt_decay	0.0
...

Logs can be found at '/mllogs'.
'drift' logs display the average values in data and collect the category probabilities.
Aggregated over time these indicators can give an idea on how big is the data drift.

Since this type of serving cannot support the multidimensional Tensor input, 
we'll have to serve it "manually" with Flask.

## Serving: Flask
`cd serve`
`sudo docker-compose up --build`

The docker will be built and served.
Model is served at 'http://localhost:5001'.
Logs can be found at '/serve/logs'.
Feel free to use the GUI or simply pass a POST request with an image.
Model returns a JSON.


## Summary

We covered the full pipeline of training and serving a model in production.
However, this could be greatly improved by migrating the model to base64 
inputs so the full pipeline would be executed via MLFlow.
This would make it much more faster and fluent process while 
working with models in production.

