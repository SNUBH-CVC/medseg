import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlflow
from ignite.engine import create_supervised_evaluator

from medseg.core.utils import import_attribute, set_mlflow_tracking_uri


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", type=str)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--multi_gpu", action="store_true")
    parser.add_argument("--mlflow_tracking_uri", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mlflow_tracking_uri is None:
        mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    else:
        mlflow_tracking_uri = args.mlflow_tracking_uri
    assert mlflow_tracking_uri is not None
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = mlflow.MlflowClient(args.mlflow_tracking_uri)
    run = client.get_run(args.run_id)
    run_id = args.run_id
    experiment_id = run.info.experiment_id
    model_uri = f"runs:/{run_id}/model"

    model = mlflow.pytorch.load_model(model_uri)

    artifact_list = mlflow.artifacts.list_artifacts(
        run_id=args.run_id, artifact_path="config"
    )
    for artifact in artifact_list:
        artifact_name = artifact.path
        if artifact_name.endswith(".py"):
            break
    else:
        raise FileNotFoundError("Configuration file missing.")
    cfg_path = f"{mlflow_tracking_uri}/{experiment_id}/{run_id}/artifacts/{artifact_name}".replace(
        "file://", ""
    )

    prepare_test_func = import_attribute(cfg_path, "prepare_test")
    items = prepare_test_func()
    evaluator_kwargs = items["evaluator_kwargs"]
    test_dataloader = items["dataloader"]
    evaluator_handlers = items["evaluator_handlers"]

    evaluator = create_supervised_evaluator(
        model, device=args.device, **evaluator_kwargs
    )

    for handler in evaluator_handlers:
        handler.attach(evaluator)
    evaluator.run(test_dataloader)


if __name__ == "__main__":
    main()
