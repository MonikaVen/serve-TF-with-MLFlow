#Dependencies

Make sure you have python, tensorflow and postgres installed:

`sudo apt install python3 python3-pip
apt install libgl1-mesa-glx
apt install postgresql postgresql-contrib postgresql-server-dev-all
sudo apt install gcc`

Install MLFlow:

`pip3 install mlflow`

Create a backend store for MLFlow:

`sudo -u postgres psql`

`CREATE DATABASE mlflow_db;
CREATE USER mlflow_user WITH ENCRYPTED PASSWORD 'mlflow';
GRANT ALL PRIVILEGES ON DATABASE mlflow_db TO mlflow_user;`

`pip3 install psycopg2-binary`

`sudo pip3 install tensorflow`

Register the model with:

`python3 serve.py`

Run MLFlow:

`mlflow server --backend-store-uri postgresql://mlflow_user:mlflow@localhost/mlflow_db --default-artifact-root ./mlruns -h 0.0.0.0 -p 8000`

Serve the model with it's artifact:

`mlflow models serve -m "./mlruns/0/b1a6b02eb7cb4537874dff1d678b2ac9/artifacts/model" -h 0.0.0.0`

Parameters are logged into 'mlruns/0/b2f1db7f53124d75af5d6ecc84d89bfb/artifacts/data.json', also displayed in MLFlow.

For the display of metrics and parameters also check MLFlow ui at:

http://localhost:8000
