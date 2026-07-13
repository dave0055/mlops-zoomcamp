#!/usr/bin/env python
# coding: utf-8

import pickle
import gc
from pathlib import Path

import numpy as np
import pandas as pd
from prefect import flow, task
from sklearn.linear_model import LinearRegression

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error

import mlflow

models_folder = Path('models')
models_folder.mkdir(exist_ok=True)


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("nyc-taxi-experiment")


def read_dataframe(filename, max_rows=None):
    base_columns = ['PULocationID', 'DOLocationID', 'trip_distance']
    time_column_candidates = [
        ('tpep_pickup_datetime', 'tpep_dropoff_datetime'),
        ('lpep_pickup_datetime', 'lpep_dropoff_datetime'),
    ]

    df = None
    pickup_col = None
    dropoff_col = None
    last_exception = None

    for candidate_pickup_col, candidate_dropoff_col in time_column_candidates:
        try:
            selected_columns = [candidate_pickup_col, candidate_dropoff_col] + base_columns
            df = pd.read_parquet(filename, columns=selected_columns)
            pickup_col = candidate_pickup_col
            dropoff_col = candidate_dropoff_col
            break
        except Exception as exc:
            last_exception = exc

    if df is None:
        raise last_exception

    print(f"Q3: Records loaded into df: {len(df)}")

    df['duration'] = df[dropoff_col] - df[pickup_col]
    df.duration = (df.duration.dt.total_seconds() / 60).astype(np.float32)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42)

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    df['trip_distance'] = df['trip_distance'].astype(np.float32)
    print(f"Q4: Records loaded into df {len(df)}")
    
    return df


def create_X(df, dv=None):
    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']

    dicts = df[categorical + numerical].to_dict(orient='records')

    if dv is None:
        dv = DictVectorizer(dtype=np.float32, sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv


@task(retries=1, retry_delay_seconds=5)
def transform_and_train(train_filename, val_filename, max_rows=None):
    df_train = read_dataframe(train_filename, max_rows=max_rows)
    X_train, dv = create_X(df_train)

    target = 'duration'
    y_train = df_train[target].to_numpy(dtype=np.float32)

    # Free training dataframe before reading validation data to lower peak memory.
    del df_train
    gc.collect()

    model = LinearRegression()
    model.fit(X_train, y_train)
    print(f"Model intercept: {model.intercept_}")

    del X_train, y_train
    gc.collect()

    df_val = read_dataframe(val_filename, max_rows=max_rows)
    X_val, _ = create_X(df_val, dv)
    y_val = df_val[target].to_numpy(dtype=np.float32)

    del df_val
    gc.collect()

    y_pred = model.predict(X_val)

    rmse = root_mean_squared_error(y_val, y_pred)
    print(f"Validation RMSE: {rmse:.4f}")

    del X_val, y_val, y_pred
    gc.collect()

    return dv, model, rmse


@task
def log_and_save_artifacts(dv, model, rmse):
    with mlflow.start_run():
        mlflow.log_metric("rmse", rmse)
        mlflow.sklearn.log_model(model, artifact_path="model")

        preprocessor_path = models_folder / "dict_vectorizer.b"
        model_path = models_folder / "lin_reg.bin"

        with open(preprocessor_path, "wb") as f_out:
            pickle.dump(dv, f_out)

        with open(model_path, "wb") as f_out:
            pickle.dump(model, f_out)

        mlflow.log_artifact(str(preprocessor_path), artifact_path="preprocessor")
        mlflow.log_artifact(str(model_path), artifact_path="model_pickle")


@flow(name="nyc-taxi-lr-training-pipeline")
def run(train_filename, val_filename, max_rows=None):
    dv, model, rmse = transform_and_train(train_filename, val_filename, max_rows=max_rows)
    log_and_save_artifacts(dv, model, rmse)

if __name__ == '__main__':
    run(
        train_filename='https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2023-01.parquet',
        val_filename='https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2023-02.parquet',
    )