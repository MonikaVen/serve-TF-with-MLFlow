import logging
import time
import datetime
import pandas as pd
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import cv2
import gc
from sklearn.preprocessing import OrdinalEncoder
import itertools
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, auc, roc_auc_score
import math
from tensorflow.keras import backend as K
import mlflow
import mlflow.tensorflow
import argparse
import sys
import os
import tempfile
from mlflow.models.signature import infer_signature

logging.basicConfig(filename='./mllogs/train.log', level=logging.DEBUG)

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

def load_data(train_path, test_path):
    train_df, test_df = None, None
    try:
        log_debug('Started loading data...')
        train_df = pd.read_parquet(train_path)
        log_debug('train.parquet loaded')
    except:
        log_error('Failed to load train.parquet.')
    try:
        test_df = pd.read_parquet(test_path)
        log_debug('test.parquet loaded')
    except:
        log_error('Failed to load test.parquet.')
    try:
        #Ensure unique
        log_debug('Ensuring unique data...')
        np.random.choice(train_df[train_df["cc3"].notna()]["cc3"].unique().tolist())
        train_df['cc3'] = train_df['cc3'].replace({None: np.random.choice(train_df[train_df["cc3"].notna()]["cc3"].unique().tolist())})
        train_df = train_df.drop_duplicates("uuid")
        train_df = train_df.reset_index(drop=True)
        log_debug('Done taking unique.')
        return train_df, test_df
    except:
        log_error('Failed to ensure unique.')       
    return train_df, test_df

def get_photos(uuids, preprocessing_fn, photos_path):
    images = []
    try:
        log_debug("Loading images.")
        for uuid in tqdm(uuids, total=len(uuids)):
            img = preprocessing_fn(cv2.resize(cv2.cvtColor(cv2.imread(f"{photos_path}/{uuid}.jpeg"), cv2.COLOR_BGR2RGB), (299, 299)))
            images.append(img)
        log_debug("Loaded images successfully.")
    except: 
        log_error("Failed to load the images.")
    return images

def transform_data(train_df, enc):
    transformed = enc.fit_transform(np.split(train_df["cc3"].values, train_df["cc3"].shape[0]))
    transformed_flattened = list(map(int, list(itertools.chain(*transformed))))
    transformed_flattened[0:10]
    train_df["cc3_encoded"] = transformed_flattened
    # print(enc.categories_)
    # print(train_df["cc3"].value_counts())
    return train_df

def define_datasets(train_df, images, BATCH_SIZE):
    CC3_COUNT = len(train_df["cc3"].unique())
    DATASET_SIZE = train_df.shape[0]
    print(CC3_COUNT, DATASET_SIZE, BATCH_SIZE)
    images_b64 = []
    for img in images:
        images_b64.append({"b64": img.decode("utf-8")})
    X = tf.data.Dataset.from_tensor_slices(images_b64)
    y_category = tf.data.Dataset.from_tensor_slices(train_df["cc3_encoded"].values)
    y_sparkling = tf.data.Dataset.from_tensor_slices(train_df["sparkling"].values)
    y_floral = tf.data.Dataset.from_tensor_slices(train_df["floral"].values)
    y_striped = tf.data.Dataset.from_tensor_slices(train_df["striped"].values)
    y = tf.data.Dataset.zip((y_category, y_sparkling, y_floral, y_striped))
    full_dataset = tf.data.Dataset.zip((X, y))
    train_size = int(0.75 * DATASET_SIZE)
    val_size = DATASET_SIZE - train_size
    train_dataset = full_dataset.take(train_size)
    val_dataset = full_dataset.skip(train_size)
    train_dataset = train_dataset.batch(BATCH_SIZE).cache().repeat()
    val_dataset = val_dataset.batch(BATCH_SIZE).cache().repeat()
    return train_dataset, val_dataset, train_size, val_size

def create_model(dp_rate):
    try:
        log_debug("Creating a model...")
        PRETRAINED_MODEL_INPUT_SHAPE = (299, 299, 3)
        pretrained_feature_extractor = tf.keras.applications.xception.Xception(include_top=False,
                                                                            weights='imagenet',
                                                                            input_tensor=None,
                                                                            input_shape=None,
                                                                            pooling="max")
        inputz = tf.keras.layers.Input(shape=PRETRAINED_MODEL_INPUT_SHAPE)
        pretrained_feature_extractor.trainable = False
        x = pretrained_feature_extractor(inputz)
        x = tf.keras.layers.Flatten()(x)
        pattern = x
        pattern = tf.keras.layers.BatchNormalization()(pattern)
        pattern = tf.keras.layers.Dropout(rate=dp_rate)(pattern)
        pattern = tf.keras.layers.Dense(units=64, activation='relu', use_bias=False)(pattern)
        pattern = tf.keras.layers.BatchNormalization()(pattern)
        pattern = tf.keras.layers.Dropout(rate=dp_rate)(pattern)
        pattern = tf.keras.layers.Dense(units=16, activation='relu', use_bias=False)(pattern)
        pattern = tf.keras.layers.BatchNormalization()(pattern)
        sparkling = tf.keras.layers.Dense(units=1, activation='sigmoid', name="sparkling_output")(pattern)
        floral = tf.keras.layers.Dense(units=1, activation='sigmoid', name="floral_output")(pattern)
        striped = tf.keras.layers.Dense(units=1, activation='sigmoid', name="striped_output")(pattern)
        shape = x
        shape = tf.keras.layers.BatchNormalization()(shape)
        shape = tf.keras.layers.Dropout(rate=dp_rate)(shape)
        shape = tf.keras.layers.Dense(units=64, activation='relu', use_bias=False)(shape)
        shape = tf.keras.layers.BatchNormalization()(shape)
        shape = tf.keras.layers.Dropout(rate=dp_rate)(shape)
        shape = tf.keras.layers.Dense(units=16, activation='relu', use_bias=False)(shape)
        shape = tf.keras.layers.BatchNormalization()(shape)
        category_predictions = tf.keras.layers.Dense(5, activation="softmax", name="category_output")(shape)
        model = tf.keras.Model(inputs=inputz, outputs=[category_predictions, sparkling, floral, striped])
        log_debug("Created model successfully.")
        return model
    except:
        log_error("Could not create a model.")
        return None

def get_metrics(train_df, val_size):
    try:
        log_debug("Getting model metrics...")
        mean_cc3 = train_df["cc3_encoded"].mode()
        np.full(val_size, mean_cc3).shape, train_df.iloc[:val_size]["cc3_encoded"].values.shape
        accuracy = accuracy_score(np.full(val_size, mean_cc3), train_df.iloc[:val_size]["cc3_encoded"].values)
        f1 = f1_score(np.full(val_size, mean_cc3), train_df.iloc[:val_size]["cc3_encoded"].values, average="macro")
        log_debug("Model metrics calculated.")
        return mean_cc3, accuracy, f1
    except:
        log_error("Failed to get model metrics.")
        return None, None, None

def get_drift(test):
    try:
        log_error("Will calculate value distribution.")
        f = open("./mllogs/drift_cat.log", "a")
        g = open("./mllogs/drift_patterns.log", "a")
        total = len(test)
        cat1 = test["category1"].value_counts()/total
        cat2 = test["category2"].value_counts()/total
        sparkling = test["sparkling"]
        floral = test["floral"]
        striped = test["striped"]
        f.write(cat1.to_string()+'\n')
        f.write(cat2.to_string()+'\n')
        g.write('sparkling\n')
        g.write(sparkling.to_string()+'\n')
        g.write('floral\n')
        g.write(floral.to_string()+'\n')
        g.write('striped\n')
        g.write(striped.to_string()+'\n')
        f.close()
        g.close()
        log_error("Calculated value distribution.")
        return 1
    except:
        log_error("Error calculating value distribution.")
        return 0
# def wbce( y_true, y_pred, weight1=1.0, weight0=0.1) :
#     epsilon = K.epsilon()
#     y_true = K.clip(y_true, epsilon, 1.-epsilon)
#     y_pred = K.clip(y_pred, epsilon, 1.-epsilon)
#     print(y_true)
#     logloss = -(y_true * K.log(y_pred) * weight1 + (1. - y_true) * K.log(1. - y_pred) * weight0 )
#     print(logloss)
#     return K.mean( logloss, axis=-1)

def compile_model(model):
    try:
        log_debug("Compiling model...")
        model.compile(optimizer=tf.keras.optimizers.Adam(),
        loss={
        "category_output": tf.keras.losses.SparseCategoricalCrossentropy(),
        "sparkling_output": "binary_crossentropy",
        "floral_output": "binary_crossentropy",
        "striped_output": "binary_crossentropy"
            },
        metrics={
            "category_output": "accuracy",
        "sparkling_output": [tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
        "floral_output": [tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
        "striped_output": [tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
            })
        log_debug("Model compiled.")
        return model
    except:
        log_error("Failed to compile a model.")
        return None

def fit_model(model, train_dataset, val_dataset, train_size, val_size, BATCH_SIZE, NUMBER_OF_EPOCHS):
    try:
        log_debug("Fitting model...")
        steps_per_epoch = math.ceil(train_size / BATCH_SIZE)
        validation_steps = math.ceil(val_size / BATCH_SIZE)
        print(steps_per_epoch, validation_steps)
        model.fit(train_dataset, 
            epochs=NUMBER_OF_EPOCHS, 
            steps_per_epoch=steps_per_epoch, 
            validation_data=val_dataset, 
            validation_steps=validation_steps,
            )
        log_debug("Fitted model successfully.")
        return model
    except:
        log_error("Failed to fit a model.")
        return None

def evaluate_model(test, model, enc, BATCH_SIZE, preprocessing_fn):
    try:
        log_debug("Evaluating models...")
        test_imgs = get_photos_base64(test["uuid"].values.tolist(), preprocessing_fn, photos_path="photos")
        predictions = model.predict(np.array(test_imgs), batch_size=BATCH_SIZE, verbose=True)
        np.sort(predictions[0][0:5, :], axis=1)[:, -1]
        np.argsort(predictions[0], axis=1)
        predictions_cc3 = np.argmax(predictions[0], axis=1)
        predictions_cc3_2 = np.argsort(predictions[0], axis=1)[:, -2]
        predictions_cc3_decoded = enc.inverse_transform(np.split(predictions_cc3, predictions_cc3.shape[0]))
        predictions_cc3_decoded_2 = enc.inverse_transform(np.split(predictions_cc3_2, predictions_cc3_2.shape[0]))
        test["category1"] = predictions_cc3_decoded
        test["category2"] = predictions_cc3_decoded_2
        test["sparkling"] = np.squeeze(predictions[1])
        test["floral"] = np.squeeze(predictions[2])
        test["striped"] = np.squeeze(predictions[3])
        test.to_parquet("./data/predictions.parquet")
        log_debug("Model evaluated successfully.")
        return test, predictions
    except:
        log_error("Failed to evaluate a model.")
        return None, None

def get_parameters(argv):
    try:
        log_debug("Getting parameters...")
        parser = argparse.ArgumentParser()
        parser.add_argument("--batch_size", default=8, type=int, help="batch size")
        parser.add_argument("--train_epochs", default=3, type=int, help="number of training epochs")
        parser.add_argument("--test_file", default="./data/test.parquet", type=str, help="test data file .parquet")
        parser.add_argument("--train_file", default="./data/train.parquet", type=str, help="train data file .parquet")
        parser.add_argument("--dp_rate", default=0.3, type=float, help="learning rate")
        parser.add_argument("--data_limit", default=50, type=int, help="limit the data being used for testing.")
        args = parser.parse_args(argv[1:])
        mlflow.log_param("batch_size", args.batch_size) 
        mlflow.log_param("train_epochs", args.train_epochs) 
        mlflow.log_param("test_file", args.test_file) 
        mlflow.log_param("train_file", args.train_file) 
        mlflow.log_param("dp_rate", args.dp_rate) 
        mlflow.log_param("data_limit", args.data_limit)
        log_debug("batch_size {}".format(args.batch_size)) 
        log_debug("train_epochs {}".format(args.train_epochs))
        log_debug("test_file {}".format(args.test_file))
        log_debug("train_file {}".format(args.train_file)) 
        log_debug("dp_rate {}".format(args.dp_rate))
        log_debug("data_limit {}".format(args.data_limit))
        log_debug("Done getting parameters.")     
        return args.batch_size, args.train_file, args.test_file, args.dp_rate, args.train_epochs, args.data_limit
    except:
        log_error("Error getting the parameters.")
        return None, None, None, None, None, None

def save_model(model, _id):
    try:
        log_debug("Saving model...")
        tf.keras.models.save_model(model, './models/'+str(_id))
        log_debug("Saved model successfully.")
        return True
    except:
        log_error("Failed to save the mmodel.")
        return False

def get_photos_base64(uuids, preprocessing_fn, photos_path):
    images = []
    try:
        for uuid in tqdm(uuids, total=len(uuids)):
            img = preprocessing_fn(cv2.resize(cv2.cvtColor(cv2.imread(f"{photos_path}/{uuid}.jpeg"), cv2.COLOR_BGR2RGB), (299, 299)))
            images.append(img)
        print("upload", len(uuids))
        log_debug("Loaded images successfully.")
        log_debug("Converting images.")      
        images64 = [base64.urlsafe_b64encode(img) for img in images]
        log_debug("Converted images successfully.")
    except: 
        log_error("Failed to load the images.")
    print(len(images64))
    return images64


def main(argv):
    try:
        log_debug("Starting mlflow run...")
        with mlflow.start_run():
            log_debug("Mlflow run started.")
            run = mlflow.active_run()
            _id = run.info.run_id
            print("Active run_id: {}".format(_id))
            mlflow.tensorflow.autolog()
            BATCH_SIZE, TRAIN_FILE, TEST_FILE, DP_RATE, NUMBER_OF_EPOCHS, DATA_LIMIT = get_parameters(argv) 
            train_df, test_df = load_data(TRAIN_FILE, TEST_FILE)
            if(len(train_df) > DATA_LIMIT):
                log_debug("Data limit is less than data length.")
                train_df = train_df[:DATA_LIMIT]
                test_df = test_df[:DATA_LIMIT]
            else:
                log_debug("Data limit is more than data length, using data length.")
                DATA_LIMIT = len(train_df)

            uuids = train_df["uuid"].values.tolist()
            preprocessing_fn = tf.keras.applications.xception.preprocess_input
            images = get_photos_base64(uuids, preprocessing_fn, "photos/")
            enc = OrdinalEncoder()
            train_df = transform_data(train_df, enc)
            train_dataset, val_dataset, train_size, val_size = define_datasets(train_df, images, BATCH_SIZE)
            model = create_model(DP_RATE)
            mean_cc3, accuracy, f1 = get_metrics(train_df, val_size)
            model = compile_model(model)
            model = fit_model(model, train_dataset, val_dataset, train_size, val_size, BATCH_SIZE, NUMBER_OF_EPOCHS)
            save_model(model, _id)
            test, predictions = evaluate_model(test_df, model, enc, BATCH_SIZE, preprocessing_fn)
            get_drift(test)
            log_debug("Ending mlflow run.")
            mlflow.end_run()
    except:
        log_error("Failed to run.")

if __name__ == "__main__":
    main(sys.argv)
