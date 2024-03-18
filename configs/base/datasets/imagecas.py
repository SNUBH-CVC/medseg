import numpy as np
from batchgenerators.transforms.color_transforms import (
    BrightnessMultiplicativeTransform, ContrastAugmentationTransform,
    GammaTransform)
from batchgenerators.transforms.noise_transforms import (
    GaussianBlurTransform, GaussianNoiseTransform)
from batchgenerators.transforms.resample_transforms import \
    SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import (MirrorTransform,
                                                           SpatialTransform)
from monai.data import DataLoader
from monai.handlers import MeanDice
from monai.inferers import SlidingWindowInferer
from monai.transforms import (Activations, AsDiscrete, Compose,
                              EnsureChannelFirstd, EnsureTyped, LoadImaged,
                              NormalizeIntensityd, RandAdjustContrastd,
                              RandAxisFlipd, RandGaussianNoised,
                              RandGaussianSmoothd, RandRotated,
                              RandScaleIntensityd, RandSpatialCropd, RandZoomd,
                              ScaleIntensityRangePercentilesd, Spacingd)

from medseg.datasets import ImageCasDataset

dataset_dir = "/data/imagecas"
num_workers = 4
batch_size = 4
roi_size = (196, 196, 128)
pixdim = [0.35, 0.35, 0.5]  # median
range_rotation = (-15.0 / 360 * 2 * np.pi, 15.0 / 360 * 2 * np.pi)
prob = 0.2
train_transform = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=pixdim),
        # https://github.com/BubblyYi/Coronary-Artery-Tracking-via-3D-CNN-Classification/blob/master/ostiapoints_train_tools/ostia_net_data_provider_aug.py
        ScaleIntensityRangePercentilesd(
            keys=["image"], lower=0.05, upper=0.95, b_min=-4000, b_max=4000
        ),
        NormalizeIntensityd(keys=["image"]),
        EnsureTyped(keys=["image", "label"]),
        RandSpatialCropd(keys=["image", "label"], roi_size=roi_size),
        # RandRotated(
        #     ["image", "label"],
        #     prob=prob,
        #     range_x=range_rotation,
        #     range_y=range_rotation,
        #     range_z=range_rotation,
        # ),
        # RandZoomd(["image", "label"], prob=prob, min_zoom=0.7, max_zoom=1.4),
        # RandGaussianNoised(keys=["image"], prob=prob),
        # RandGaussianSmoothd(keys=["image"], prob=prob),
        # RandScaleIntensityd(keys=["image"], factors=0.3, prob=prob),
        # RandAxisFlipd(["image", "label"], prob=prob),
        # RandAdjustContrastd(keys=["image"], prob=prob),
        # RandSpatialCropd(keys=["image", "label"], roi_size=roi_size),
        # EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        # EnsureTyped(keys=["image", "label"], data_type="numpy"),
        # https://github.com/PierreRouge/Cascaded-U-Net-for-vessel-segmentation
        # adaptor(
        #     SpatialTransform(
        #         roi_size,
        #         patch_center_dist_from_border=None,
        #         do_elastic_deform=False,
        #         alpha=(0, 0),
        #         sigma=(0, 0),
        #         do_rotation=True,
        #         angle_x=rotation_range,
        #         angle_y=rotation_range,
        #         angle_z=rotation_range,
        #         p_rot_per_axis=1,
        #         do_scale=True,
        #         scale=(0.9, 1.1),
        #         border_mode_data="constant",
        #         border_cval_data=0,
        #         order_data=3,
        #         border_mode_seg="constant",
        #         border_cval_seg=-1,
        #         order_seg=1,
        #         random_crop=False,
        #         p_el_per_sample=0,
        #         p_scale_per_sample=0.2,
        #         p_rot_per_sample=0.2,
        #         independent_scale_for_each_axis=False,
        #         data_key="image",
        #         label_key=("label"),
        #     ),
        #     {"image": "image", "label": "label"},
        # ),
        # adaptor(
        #     GaussianNoiseTransform(p_per_sample=0.1, data_key="image"),
        #     {"image": "image", "label": "label"},
        # ),
        # adaptor(
        #     GaussianBlurTransform(
        #         (0.8, 1.0),
        #         different_sigma_per_channel=True,
        #         p_per_sample=0.2,
        #         p_per_channel=0.5,
        #         data_key="image",
        #     ),
        #     {"image": "image", "label": "label"},
        # ),
        # adaptor(
        #     BrightnessMultiplicativeTransform(
        #         multiplier_range=(0.9, 1.1), p_per_sample=0.15, data_key="image"
        #     ),
        #     {"image": "image", "label": "label"},
        # ),
        # adaptor(
        #     ContrastAugmentationTransform(p_per_sample=0.1, data_key="image"),
        #     {"image": "image", "label": "label"},
        # ),
        # adaptor(
        #     SimulateLowResolutionTransform(
        #         zoom_range=(0.8, 1),
        #         per_channel=True,
        #         p_per_channel=0.5,
        #         order_downsample=0,
        #         order_upsample=3,
        #         p_per_sample=0.25,
        #         ignore_axes=None,
        #         data_key="image",
        #     ),
        #     {"image": "image", "label": "label"},
        # ),
        # adaptor(
        #     GammaTransform(
        #         (0.9, 1.1),
        #         True,
        #         True,
        #         retain_stats=True,
        #         p_per_sample=0.1,
        #         data_key="image",
        #     ),
        #     {"image": "image", "label": "label"},
        # ),
        # adaptor(
        #     MirrorTransform((0, 1, 2), data_key="image", label_key=("label")),
        #     {"image": "image", "label": "label"},
        # ),
        # SqueezeDimd(keys=["image", "label"], dim=0),
    ]
)
train_dataset = ImageCasDataset(
    dataset_dir=dataset_dir,
    mode="train",
    transform=train_transform,
    cache_rate=0.0,
)
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
)
eval_transform = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=pixdim),
        ScaleIntensityRangePercentilesd(
            keys=["image"], lower=0.05, upper=0.95, b_min=-4000, b_max=4000
        ),
        NormalizeIntensityd(keys=["image"]),
        EnsureTyped(keys=["image", "label"]),
    ]
)
eval_post_pred_transforms = Compose(
    [Activations(sigmoid=True), AsDiscrete(threshold=0.5)]
)
eval_post_label_transforms = Compose([AsDiscrete(threshold=0.5)])

val_dataset = ImageCasDataset(
    dataset_dir=dataset_dir,
    mode="validation",
    transform=eval_transform,
    cache_rate=0.0,
)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

test_dataset = ImageCasDataset(
    dataset_dir=dataset_dir,
    mode="test",
    transform=eval_transform,
    cache_rate=0.0,
)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

inferer = SlidingWindowInferer(roi_size=roi_size, sw_batch_size=4, overlap=0.25)
key_metric_name = "mean_dice"
metrics = {key_metric_name: MeanDice()}
