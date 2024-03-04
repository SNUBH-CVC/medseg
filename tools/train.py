import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medseg.core.trainer import SupervisedTrainWrapper
from medseg.core.utils import import_attribute, set_mlflow_tracking_uri


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_path", type=str)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--multi_gpu", action="store_true")
    parser.add_argument("--mlflow_tracking_uri", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    set_mlflow_tracking_uri(args.mlflow_tracking_uri)

    prepare_train_func = import_attribute(args.cfg_path, "prepare_train")
    items = prepare_train_func()

    trainer = SupervisedTrainWrapper(args.cfg_path, args.device, **items)
    trainer.run()


if __name__ == "__main__":
    main()
