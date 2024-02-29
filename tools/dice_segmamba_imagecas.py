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
from monai.transforms import (Activationsd, AsDiscreted, Compose, EnsureTyped,
                              KeepLargestConnectedComponentd, LoadImaged,
                              NormalizeIntensityd, RandSpatialCropd,
                              ScaleIntensityRangePercentilesd)

from medseg.dataset import ImageCASDataset
from medseg.model.segmamba import SegMamba
from medseg.core.utils import get_basename_wo_ext


def main():
    monai.config.print_config()

    # set hyperparameters
    dataset_dir = "/data/imagecas"
    log_root_dir = "./runs"
    batch_size = 16
    max_epochs = 100
    num_workers = 4
    roi_size = (128, 128, 64)
    multi_gpu = True
    tensorboard_log_name = f"{get_basename_wo_ext(__file__)}-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    log_dir = os.path.join(log_root_dir, tensorboard_log_name)
    os.makedirs(log_dir)
    copyfile(__file__, os.path.join(log_dir, os.path.basename(__file__)))

    # set root log level to INFO and init a train logger, will be used in `StatsHandler`
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    get_logger("train_log")

    # define transforms for image and segmentation
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=0.95),
            NormalizeIntensityd(keys=["image"]),
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
            LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=0.95),
            NormalizeIntensityd(keys=["image"]),
            EnsureTyped(keys=["image", "label"]),
        ]
    )

    # create a training data loader
    train_ds = ImageCASDataset(
        dataset_dir=dataset_dir,
        section="training",
        transform=train_transforms,
        cache_rate=0.0,
        val_frac=0.2,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_ds = ImageCASDataset(
        dataset_dir=dataset_dir,
        section="validation",
        transform=val_transforms,
        cache_rate=0.0,
        val_frac=0.2,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=num_workers
    )

    # create network, optimizer and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = SegMamba(
        in_chans=1, out_chans=1, depths=[2, 2, 2, 2], feat_size=[48, 96, 192, 384]
    ).to(device)
    if multi_gpu and torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs!")
        net = nn.DataParallel(net)
    loss_function = monai.losses.DiceLoss(sigmoid=True)
    optimizer = torch.optim.Adam(net.parameters(), 1e-4, weight_decay=1e-5)
    # lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=max_epochs)
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
        # EarlyStopHandler(
        #     trainer=None,
        #     patience=2,
        #     score_function=lambda x: x.state.metrics["val_mean_dice"],
        # ),
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
        inferer=SlidingWindowInferer(roi_size=roi_size, sw_batch_size=4, overlap=0.25),
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
        # EarlyStopHandler(
        #     trainer=None,
        #     patience=20,
        #     score_function=lambda x: -x.state.output[0]["loss"],
        #     epoch_level=False,
        # ),
        # LrScheduleHandler(lr_scheduler=lr_scheduler, print_lr=True),
        ValidationHandler(validator=evaluator, interval=2, epoch_level=True),
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
    # val_handlers[0].set_trainer(trainer=trainer)
    # train_handlers[0].set_trainer(trainer=trainer)
    trainer.run()


if __name__ == "__main__":
    main()
