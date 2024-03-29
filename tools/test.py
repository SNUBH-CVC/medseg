import argparse
import os

import hydra
import mlflow
import omegaconf
from monai.data import DataLoader
from monai.transforms import (Compose, EnsureChannelFirstd,
                              EnsureTyped, LoadImaged, NormalizeIntensityd,
                              SaveImage, ScaleIntensityRangePercentilesd,
                              Spacingd)

from medseg.core.utils import set_mlflow_tracking_uri
from medseg.datasets import ImageCasDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("mlflow_tracking_uri", type=str)
    parser.add_argument("run_id", type=str)
    parser.add_argument("output_dir", type=str, default="outputs")
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()

    set_mlflow_tracking_uri(args.mlflow_tracking_uri)
    client = mlflow.MlflowClient(args.mlflow_tracking_uri)
    run = client.get_run(args.run_id)

    # get hydra config from artifact path
    hydra_output_subdir = run.data.params["hydra_output_subdir"]
    hydra_config_path = os.path.join(hydra_output_subdir, "config.yaml")
    cfg = omegaconf.OmegaConf.load(hydra_config_path)

    dataset = hydra.utils.instantiate(cfg.dataset)
    inferer = dataset["inferer"]
    eval_post_pred_transforms = dataset["eval_post_pred_transforms"]

    # load model
    model_uri = f"runs:/{args.run_id}/model"
    model = mlflow.pytorch.load_model(model_uri).to(args.device)
    spacing = [0.34960938, 0.34960938, 0.5]

    transforms = Compose(
        [
            LoadImaged(keys=["image"], image_only=False),
            EnsureChannelFirstd(keys=["image"]),
            Spacingd(keys=["image"], pixdim=spacing),
            ScaleIntensityRangePercentilesd(
                keys=["image"], lower=0.05, upper=0.95, b_min=-4000, b_max=4000
            ),
            NormalizeIntensityd(keys=["image"]),
            EnsureTyped(keys=["image"]),
        ]
    )

    # https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/torch/unet_evaluation_dict.py#L97
    saver = SaveImage(
        resample=True,
        output_dir="./outputs",
        output_ext=".nii.gz",
        output_postfix="seg",
        separate_folder=False,
    )

    test_dataset = ImageCasDataset(
        "/data/public/raw_coco_format/imagecas",
        "test.json",
        "images",
        mode="test",
        mask_dirname="masks",
        transform=transforms,
    )
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    for i, batch in enumerate(test_dataloader):
        spacing = batch["meta"]["spacing"][0]
        image = batch["image"].to(args.device)
        output = inferer(image, model)
        output = eval_post_pred_transforms(output)
        saver(output[0])
        break


if __name__ == "__main__":
    main()
