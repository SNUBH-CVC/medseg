import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlflow

from medseg.core.tester import SupervisedTestWrapper
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
    set_mlflow_tracking_uri(args.mlflow_tracking_uri)

    model_uri = f"runs:/{args.run_id}/model"
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
    cfg_path = (
        f"{args.mlflow_tracking_uri}/0/{args.run_id}/artifacts/{artifact_name}".replace(
            "file://", ""
        )
    )

    prepare_test_func = import_attribute(cfg_path, "prepare_test")
    items = prepare_test_func()
    tester = SupervisedTestWrapper(model, args.device, **items)
    tester.run()


if __name__ == "__main__":
    main()
