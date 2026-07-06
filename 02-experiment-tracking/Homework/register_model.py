import os
import pickle
import json
from urllib import request, error
import click
import mlflow

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

HPO_EXPERIMENT_NAME = "random-forest-hyperopt"
EXPERIMENT_NAME = "random-forest-best-models"
REGISTERED_MODEL_NAME = "nyc-taxi-rf-best-model"
RF_PARAMS = ['max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'random_state']

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(EXPERIMENT_NAME)


def _post_json(url, payload):
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        response_body = exc.read().decode("utf-8")
        try:
            return exc.code, json.loads(response_body)
        except json.JSONDecodeError:
            return exc.code, {"message": response_body}


def _register_model_via_rest(client, run_id, model_name):
    tracking_uri = mlflow.get_tracking_uri().rstrip("/")
    if not tracking_uri.startswith("http"):
        raise RuntimeError("Tracking URI must be an HTTP endpoint for REST registration fallback")

    create_model_url = f"{tracking_uri}/api/2.0/mlflow/registered-models/create"
    status_code, response = _post_json(create_model_url, {"name": model_name})
    if status_code not in (200, 400):
        raise RuntimeError(f"Failed to create registered model: {response}")

    # MLflow model version create API expects the concrete artifact source URI.
    run = client.get_run(run_id)
    source = f"{run.info.artifact_uri}/model"
    create_version_url = f"{tracking_uri}/api/2.0/mlflow/model-versions/create"
    status_code, response = _post_json(
        create_version_url,
        {"name": model_name, "source": source, "run_id": run_id},
    )
    if status_code != 200:
        raise RuntimeError(f"Failed to create model version: {response}")

    return response["model_version"]["version"]


def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def train_and_log_model(data_path, params):
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

    with mlflow.start_run():
        new_params = {}
        for param in RF_PARAMS:
            new_params[param] = int(params[param])

        rf = RandomForestRegressor(**new_params)
        rf.fit(X_train, y_train)

        mlflow.sklearn.log_model(rf, artifact_path="model")

        # Evaluate model on the validation and test sets
        val_rmse = mean_squared_error(y_val, rf.predict(X_val), squared=False)
        mlflow.log_metric("val_rmse", val_rmse)
        test_rmse = mean_squared_error(y_test, rf.predict(X_test), squared=False)
        mlflow.log_metric("test_rmse", test_rmse)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--top_n",
    default=5,
    type=int,
    help="Number of top models that need to be evaluated to decide which one to promote"
)
def run_register_model(data_path: str, top_n: int):

    client = MlflowClient()

    # Retrieve the top_n model runs and log the models
    experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(
            f"Experiment '{HPO_EXPERIMENT_NAME}' was not found. "
            "Run hpo.py first to generate candidate runs."
        )

    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.rmse ASC"]
    )
    for run in runs:
        train_and_log_model(data_path=data_path, params=run.data.params)

    # Select the model with the lowest test RMSE
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.test_rmse ASC"]
    )[0]

    # Register the best model
    model_uri = f"runs:/{best_run.info.run_id}/model"
    try:
        result = mlflow.register_model(
            model_uri=model_uri,
            name=REGISTERED_MODEL_NAME
        )
        version = result.version
    except Exception as exc:
        if "preview/mlflow/registered-models/create" not in str(exc):
            raise
        version = _register_model_via_rest(
            client=client,
            run_id=best_run.info.run_id,
            model_name=REGISTERED_MODEL_NAME,
        )

    print(f"Registered model '{REGISTERED_MODEL_NAME}' from run {best_run.info.run_id}")
    print(f"Model version: {version}")


if __name__ == '__main__':
    run_register_model()