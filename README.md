sudo apt install python3 python3-pip
pip3 install mlflow
apt install libgl1-mesa-glx
apt install postgresql postgresql-contrib postgresql-server-dev-all


sudo -u postgres psql

CREATE DATABASE mlflow_db;
CREATE USER mlflow_user WITH ENCRYPTED PASSWORD 'mlflow';
GRANT ALL PRIVILEGES ON DATABASE mlflow_db TO mlflow_user;

sudo apt install gcc
pip install psycopg2-binary

sudo pip3 install tensorflow

mlflow server --backend-store-uri postgresql://mlflow_user:mlflow@localhost/mlflow_db --default-artifact-root ./mlruns -h 0.0.0.0 -p 8000

mlflow models serve -m "./mlruns/0/b1a6b02eb7cb4537874dff1d678b2ac9/artifacts/model" -h 0.0.0.0
