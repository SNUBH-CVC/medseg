import math
import os

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from ignite.engine import Events, create_supervised_evaluator
from monai.data import DataLoader, decollate_batch
from monai.handlers import MeanDice
from monai.inferers import SlidingWindowInferer
from monai.metrics import DiceMetric
from monai.transforms import (Activations, AsDiscrete, Compose,
                              EnsureChannelFirstd, EnsureTyped, LoadImaged,
                              NormalizeIntensityd,
                              ScaleIntensityRangePercentilesd)
from monai.visualize import blend_images

from medseg.dataset import ImageCASDataset

mlflow.set_tracking_uri("file:///data/mlruns")
logged_model = "runs:/a284c8dac16448d8b23a54ed5d728140/model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = mlflow.pytorch.load_model(logged_model).to(device)

val_transform = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRangePercentilesd(
            keys=["image"], lower=0.05, upper=0.95, b_min=-4000, b_max=4000
        ),
        NormalizeIntensityd(keys=["image"]),
        EnsureTyped(keys=["image", "label"]),
    ]
)
val_dataset = ImageCASDataset(
    dataset_dir="/data/imagecas",
    section="validation",
    transform=val_transform,
    cache_rate=0.0,
    val_frac=0.2,
)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
roi_size = (128, 128, 64)
sw_batch_size = 4
val_post_pred_transforms = Compose(
    [Activations(sigmoid=True), AsDiscrete(threshold=0.5)]
)
val_post_label_transforms = Compose([AsDiscrete(threshold=0.5)])
dice_metric = DiceMetric(include_background=True, reduction="mean")


def draw_images(images, num_rows, num_cols, save_path):
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3))
    for i, ax in enumerate(axes.flat):
        # Check if there are more images than subplots
        if i < len(images):
            ax.imshow(images[i])  # Draw image on subplot
        ax.axis("off")  # Hide axes
    plt.tight_layout()  # Adjust layout
    plt.savefig(save_path)


num_rows = 10
output_dir = "./outputs"
inferer = SlidingWindowInferer(roi_size=roi_size, sw_batch_size=4, overlap=0.25)
evaluator = create_supervised_evaluator(
    model,
    metrics={"mean_dice": MeanDice()},
    device=device,
    prepare_batch=lambda batch, device, non_blocking: (
        batch["image"].to(device),
        batch["label"].to(device),
    ),
    output_transform=lambda x, y, y_pred: (
        [val_post_pred_transforms(i) for i in y_pred],
        [val_post_label_transforms(i) for i in y],
    ),
    model_fn=lambda model, x: inferer(x, model),
)


def save_validation_result(engine):
    batch = engine.state.batch
    _id = batch["id"][0]
    images = batch["image"]
    outputs, labels = engine.state.output
    single_dice = dice_metric(outputs, labels)
    print(single_dice)
    num_slices = images.shape[-1]
    correct_blend = blend_images(
        image=images[0].cpu().numpy(),
        label=labels[0].cpu().numpy(),
        alpha=0.5,
        cmap="spring",
        rescale_arrays=False,
    )
    predict_blend = blend_images(
        image=images[0].cpu().numpy(),
        label=outputs[0].cpu().numpy(),
        alpha=0.5,
        cmap="spring",
        rescale_arrays=False,
    )
    num_cols = math.ceil(num_slices / num_rows)
    draw_images(
        np.transpose(correct_blend, (3, 1, 2, 0)),
        num_rows,
        num_cols,
        os.path.join(output_dir, f"{_id}-correct.png"),
    )
    draw_images(
        np.transpose(predict_blend, (3, 1, 2, 0)),
        num_rows,
        num_cols,
        os.path.join(output_dir, f"{_id}-predict.png"),
    )


def finish_validation(engine):
    print(engine.state.metric_details)


evaluator.add_event_handler(Events.ITERATION_COMPLETED, save_validation_result)
evaluator.add_event_handler(Events.EPOCH_COMPLETED, finish_validation)
evaluator.run(val_dataloader)
