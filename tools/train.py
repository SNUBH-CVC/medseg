# https://github.com/Project-MONAI/tutorials/blob/main/modules/engines/unet_training_dict.py

import datetime
import logging
import os
import sys
from shutil import copyfile

import monai
import torch
import torch.nn as nn
from monai.apps import get_logger
from monai.data import DataLoader
from monai.engines import SupervisedEvaluator, SupervisedTrainer
from monai.handlers import (CheckpointSaver, EarlyStopHandler,
                            LrScheduleHandler, MeanDice, StatsHandler,
                            TensorBoardImageHandler, TensorBoardStatsHandler,
                            ValidationHandler, from_engine)
from monai.inferers import SimpleInferer, SlidingWindowInferer
from monai.transforms import (Activationsd, AsDiscreted, Compose,
                              EnsureChannelFirstd, EnsureTyped,
                              KeepLargestConnectedComponentd, LoadImaged,
                              RandSpatialCropd, ScaleIntensityd,
                              ScaleIntensityRanged)

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from medseg.dataset import ImageCASDataset
from medseg.model.segmamba import SegMamba


def main():
    monai.config.print_config()

    # set hyperparameters
    data_root_dir = "/data"
    log_root_dir = "./runs"
    batch_size = 8
    max_epochs = 100
    num_workers = 4
    roi_size = (128, 128, 64)
    multi_gpu = True
    tensorboard_log_name = (
        f"segmamba-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    )
    log_dir = os.path.join(log_root_dir, tensorboard_log_name)
    os.makedirs(log_dir)
    copyfile(__file__, os.path.join(log_dir, os.path.basename(__file__)))

    # set root log level to INFO and init a train logger, will be used in `StatsHandler`
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    get_logger("train_log")

    # define transforms for image and segmentation
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            # From ImageCAS original code
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-4000,
                a_max=4000,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            # RandScaleIntensityd(keys=["image"], factors=0.1, prob=1.0),
            # RandShiftIntensityd(keys=["image"], offsets=0.1, prob=1.0),
            EnsureTyped(keys=["image", "label"]),
            RandSpatialCropd(
                keys=["image", "label"], roi_size=roi_size, random_size=False
            ),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityd(keys="image"),
            EnsureTyped(keys=["image", "label"]),
        ]
    )

    # create a training data loader
    train_ds = ImageCASDataset(
        root_dir=data_root_dir,
        section="training",
        transform=train_transforms,
        cache_rate=0.0,
        val_frac=0.5,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_ds = ImageCASDataset(
        root_dir=data_root_dir,
        section="validation",
        transform=val_transforms,
        cache_rate=0.0,
        val_frac=0.5,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=num_workers
    )

    # create network, optimizer and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = SegMamba(
        in_chans=1, out_chans=1, depths=[2, 2, 2, 2], feat_size=[48, 96, 192, 384]
    )
    if multi_gpu and torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs!")
        net = nn.DataParallel(net)
    net.to(device)
    loss_function = monai.losses.DiceLoss(sigmoid=True)
    optimizer = torch.optim.Adam(net.parameters(), 1e-3)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    val_post_transforms = Compose(
        [
            EnsureTyped(keys="pred"),
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", threshold=0.5),
            KeepLargestConnectedComponentd(keys="pred", applied_labels=[1]),
        ]
    )
    val_handlers = [
        # apply “EarlyStop” logic based on the validation metrics
        EarlyStopHandler(
            trainer=None,
            patience=2,
            score_function=lambda x: x.state.metrics["val_mean_dice"],
        ),
        # doesn't calculate validation loss: https://github.com/Project-MONAI/MONAI/discussions/3786 
        StatsHandler(name="validation_log", output_transform=lambda x: None),
        TensorBoardStatsHandler(log_dir=log_dir, output_transform=lambda x: None),
        TensorBoardImageHandler(
            log_dir=log_dir,
            batch_transform=from_engine(["image", "label"]),
            output_transform=from_engine(["pred"]),
        ),
        CheckpointSaver(save_dir=log_dir, save_dict={"net": net}, save_key_metric=True),
    ]
    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=val_loader,
        network=net,
        inferer=SlidingWindowInferer(
            roi_size=(96, 96, 96), sw_batch_size=4, overlap=0.25
        ),
        postprocessing=val_post_transforms,
        key_val_metric={
            "val_mean_dice": MeanDice(
                include_background=True, output_transform=from_engine(["pred", "label"])
            )
        },
        val_handlers=val_handlers,
        amp=True,
    )
    train_post_transforms = Compose(
        [
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", threshold=0.5),
            KeepLargestConnectedComponentd(keys="pred", applied_labels=[1]),
        ]
    )
    train_handlers = [
        # apply “EarlyStop” logic based on the loss value, use “-” negative value because smaller loss is better
        EarlyStopHandler(
            trainer=None,
            patience=20,
            score_function=lambda x: -x.state.output[0]["loss"],
            epoch_level=False,
        ),
        LrScheduleHandler(lr_scheduler=lr_scheduler, print_lr=True),
        ValidationHandler(validator=evaluator, interval=1, epoch_level=True),
        # use the logger "train_log" defined at the beginning of this program
        StatsHandler(
            name="train_log",
            tag_name="train_loss",
            output_transform=from_engine(["loss"], first=True),
        ),
        TensorBoardStatsHandler(
            log_dir=log_dir,
            tag_name="train_loss",
            output_transform=from_engine(["loss"], first=True),
        ),
    ]
    trainer = SupervisedTrainer(
        device=device,
        max_epochs=max_epochs,
        train_data_loader=train_loader,
        network=net,
        optimizer=optimizer,
        loss_function=loss_function,
        inferer=SimpleInferer(),
        postprocessing=train_post_transforms,
        key_train_metric={
            "train_acc": MeanDice(output_transform=from_engine(["pred", "label"]))
        },
        train_handlers=train_handlers,
        # if no FP16 support in GPU or PyTorch version < 1.6, will not enable AMP training
        amp=True,
    )
    # set initialized trainer for "early stop" handlers
    val_handlers[0].set_trainer(trainer=trainer)
    train_handlers[0].set_trainer(trainer=trainer)
    trainer.run()


if __name__ == "__main__":
    main()
