import os
import numpy as np
import math
import torch
import matplotlib.pyplot as plt
import mlflow
from monai.inferers import sliding_window_inference
from monai.data import DataLoader, decollate_batch
from monai.metrics import DiceMetric
from monai.visualize import blend_images
from monai.transforms import (Compose, AsDiscrete,
                              EnsureChannelFirstd, EnsureTyped, LoadImaged,
                              NormalizeIntensityd, Activations,
                              ScaleIntensityRangePercentilesd)
from medseg.dataset import ImageCASDataset

mlflow.set_tracking_uri("file:///data/mlruns")
logged_model = 'runs:/3b81f6641e834f5b851b7b4fb357a2de/model'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = mlflow.pytorch.load_model(logged_model).to(device)
model.eval()

val_transform = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=0.95, b_min=-4000, b_max=4000),
        NormalizeIntensityd(keys=["image"]),
        EnsureTyped(keys=["image", "label"]),
    ]
)
val_dataset = ImageCASDataset(
    dataset_dir="/data/imagecas", section="validation", transform=val_transform, cache_rate=0.0, val_frac=0.2
)
val_dataloader = DataLoader(
    val_dataset, batch_size=1, shuffle=False, num_workers=1
)
roi_size = (128, 128, 64)
sw_batch_size = 4
post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
post_label = Compose([AsDiscrete(threshold=0.5)])
dice_metric = DiceMetric(
    include_background=True, reduction="mean"
)

def draw_images(images, num_rows, num_cols, save_path):
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3))
    for i, ax in enumerate(axes.flat):
        # Check if there are more images than subplots
        if i < len(images):
            ax.imshow(images[i])  # Draw image on subplot
        ax.axis('off')  # Hide axes
    plt.tight_layout()  # Adjust layout
    plt.savefig(save_path)


num_rows = 10
output_dir = "./outputs"
with torch.no_grad():
    # TODO: validation batch size 1 이상인 경우 대응
    for val_data in val_dataloader:
        val_images, val_labels = val_data["image"].to(device), val_data[
            "label"
        ].to(device)
        val_id = val_data["id"]
        val_outputs = sliding_window_inference(
            val_images, roi_size, sw_batch_size, model
        )
        val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
        val_labels = [post_label(i) for i in decollate_batch(val_labels)]
        metric = dice_metric(y_pred=val_outputs, y=val_labels)

        # visualize
        num_slices = val_images.shape[-1]
        correct_blend = blend_images(image=val_images[0].cpu().numpy(), label=val_labels[0].cpu().numpy(), alpha=0.5, cmap="spring", rescale_arrays=False)
        predict_blend = blend_images(image=val_images[0].cpu().numpy(), label=val_outputs[0].cpu().numpy(), alpha=0.5, cmap="spring", rescale_arrays=False)
        num_cols = math.ceil(num_slices / num_rows)
        draw_images(np.transpose(correct_blend, (3, 1, 2, 0)), num_rows, num_cols, os.path.join(output_dir, f"{val_id}-correct.png"))
        draw_images(np.transpose(predict_blend, (3, 1, 2, 0)), num_rows, num_cols, os.path.join(output_dir, f"{val_id}-predict.png"))

    result = dice_metric.aggregate().item()
    print(result)
