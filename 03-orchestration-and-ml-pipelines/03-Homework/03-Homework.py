#!/usr/bin/env python
# coding: utf-8

import os
import pickle
from pathlib import Path

# Avoid a UnicodeEncodeError on Windows consoles: mlflow prints an emoji
# in its run-finished message before the cp1250/cp1252 codec can handle it.
os.environ.setdefault("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT", "1")

import pandas as pd
import requests
import prefect
from prefect import flow, task
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import DictVectorizer

import mlflow
from mlflow.models import Model

models_folder = Path('models')
models_folder.mkdir(exist_ok=True)

data_folder = Path('data')
data_folder.mkdir(exist_ok=True)


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("nyc-taxi-experiment")


def _download(url):
    local_path = data_folder / Path(url).name
    if not local_path.exists():
        with requests.get(url, stream=True) as resp:
            resp.raise_for_status()
            with open(local_path, "wb") as f_out:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    f_out.write(chunk)
    return local_path


@task(retries=1, retry_delay_seconds=5)
def read_dataframe(filename):
    local_path = _download(filename)
    df = pd.read_parquet(local_path)
    print(f"Q3: Records loaded into df: {len(df)}")

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    print(f"Q4: Records after preparation: {len(df)}")

    return df


def create_X(df, dv=None):
    categorical = ['PULocationID', 'DOLocationID']
    dicts = df[categorical].to_dict(orient='records')

    if dv is None:
        dv = DictVectorizer()
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv


@task
def train_model(df):
    X_train, dv = create_X(df)

    target = 'duration'
    y_train = df[target].values

    model = LinearRegression()
    model.fit(X_train, y_train)
    print(f"Q5: Model intercept: {model.intercept_}")

    return dv, model


@task
def log_and_save_artifacts(dv, model):
    with mlflow.start_run() as run:
        model_info = mlflow.sklearn.log_model(model, artifact_path="model")

        preprocessor_path = models_folder / "dict_vectorizer.b"
        with open(preprocessor_path, "wb") as f_out:
            pickle.dump(dv, f_out)

        mlflow.log_artifact(str(preprocessor_path), artifact_path="preprocessor")

        model_size_bytes = Model.load(model_info.model_uri).model_size_bytes
        print(f"Q6: model_size_bytes: {model_size_bytes}")

        return run.info.run_id


@flow(name="nyc-taxi-lr-training-pipeline")
def run(filename):
    print(f"Q1: Orchestrator: Prefect")
    print(f"Q2: Orchestrator version: {prefect.__version__}")

    df = read_dataframe(filename)
    dv, model = train_model(df)
    run_id = log_and_save_artifacts(dv, model)
    print(f"MLflow run_id: {run_id}")
    return run_id


if __name__ == '__main__':
    run(filename='https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet')
