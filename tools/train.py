import logging
import os
import sys

import mlflow
import torch
import torch.nn as nn
from ignite.engine import (create_supervised_evaluator,
                           create_supervised_trainer)
from monai.data import DataLoader
from monai.handlers import MeanDice
from monai.inferers import SlidingWindowInferer
from monai.losses.dice import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets.unet import UNet
from monai.transforms import (Activations, AsDiscrete, Compose,
                              EnsureChannelFirstd, EnsureTyped, LoadImaged,
                              NormalizeIntensityd, RandCropByPosNegLabeld,
                              RandScaleIntensityd, RandShiftIntensityd,
                              ScaleIntensityRangePercentilesd)

from medseg.core.trainer import MedSegSupervisedTrainer
from medseg.dataset import ImageCASDataset

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


# hyperparameters
root_dir = "/data/imagecas"
multi_gpu = True
roi_size = (128, 128, 64)
max_epochs = 200
val_frac = 0.2
batch_size = 16
run_name = "imagecas"
num_workers = 4

mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
if mlflow_tracking_uri is None:
    mlflow_tracking_uri = "file:///data/mlruns"
mlflow.set_tracking_uri(mlflow_tracking_uri)

# transform
train_transform = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        # https://github.com/BubblyYi/Coronary-Artery-Tracking-via-3D-CNN-Classification/blob/master/ostiapoints_train_tools/ostia_net_data_provider_aug.py
        ScaleIntensityRangePercentilesd(
            keys=["image"], lower=0.05, upper=0.95, b_min=-4000, b_max=4000
        ),
        NormalizeIntensityd(keys=["image"]),
        # RandScaleIntensityd(keys=["image"], factors=0.1, prob=1.0),
        # RandShiftIntensityd(keys=["image"], offsets=0.1, prob=1.0),
        EnsureTyped(keys=["image", "label"]),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=roi_size,
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
        ),
    ]
)
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

# create a training data loader
train_dataset = ImageCASDataset(
    dataset_dir=root_dir,
    section="training",
    transform=train_transform,
    cache_rate=0.0,
    val_frac=val_frac,
)
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
)
val_dataset = ImageCASDataset(
    dataset_dir=root_dir,
    section="validation",
    transform=val_transform,
    cache_rate=0.0,
    val_frac=val_frac,
)
val_dataloader = DataLoader(
    val_dataset, batch_size=1, shuffle=False, num_workers=num_workers
)
dice_metric = DiceMetric(include_background=False, reduction="mean")
val_post_pred_transforms = Compose(
    [Activations(sigmoid=True), AsDiscrete(threshold=0.5)]
)
val_post_label_transforms = Compose([AsDiscrete(threshold=0.5)])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UNet_metadata = {
    "spatial_dims": 3,
    "in_channels": 1,
    "out_channels": 1,
    "channels": (16, 32, 64, 128, 256),
    "strides": (2, 2, 2, 2),
    "num_res_units": 2,
}
model = UNet(**UNet_metadata).to(device)
if multi_gpu and torch.cuda.device_count() > 1:
    logger.info(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)
model.to(device)
loss_function = DiceLoss(sigmoid=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-5)

inferer = SlidingWindowInferer(roi_size=roi_size, sw_batch_size=4, overlap=0.25)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.9)

trainer = create_supervised_trainer(
    model,
    optimizer,
    loss_function,
    device=device,
    prepare_batch=lambda batch, device, non_blocking: (
        batch["image"].to(device),
        batch["label"].to(device),
    ),
)
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

medseg_trainer = MedSegSupervisedTrainer(
    run_name, __file__, model, trainer, evaluator, train_dataloader, val_dataloader
)
medseg_trainer.run(max_epochs=max_epochs)
