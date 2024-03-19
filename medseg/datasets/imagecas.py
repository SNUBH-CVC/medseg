import json
import os
import sys

from monai.data import CacheDataset
from monai.transforms import Randomizable

from .utils import PanopticCOCO


# https://github.com/Project-MONAI/tutorials/blob/main/modules/public_datasets.ipynb
class ImageCasDataset(Randomizable, CacheDataset):
    dataset_name = "imagecas"

    def __init__(
        self,
        dataset_dir,
        annotation_filename,
        img_dirname,
        mode,
        transform=None,
        download=False,
        cache_num=sys.maxsize,
        cache_rate=0.0,
        num_workers=0,
        mask_dirname=None,
        skeleton_dirname=None,
        splits_path=None,
        k=1,
    ):
        self.coco = PanopticCOCO(os.path.join(dataset_dir, annotation_filename))
        self.mode = mode
        assert 1 <= k <= 4
        if download:
            raise ValueError("Download the dataset manually.")

        img_dir = os.path.join(dataset_dir, img_dirname)
        use_mask = mask_dirname is not None
        use_skeleton = skeleton_dirname is not None
        assert use_mask or use_skeleton
        if use_mask:
            mask_dir = os.path.join(dataset_dir, mask_dirname)
        if use_skeleton:
            skeleton_dir = os.path.join(dataset_dir, skeleton_dirname)

        self.data_list = []
        if splits_path is not None:
            with open(splits_path, "r") as f:
                split_d = json.load(splits_path)
            assert len(split_d) == 5
            img_ids = split_d[k][mode]
        else:
            img_ids = self.coco.getImgIds()

        for _id in img_ids:
            img_info = self.coco.loadImgs(_id)[0]
            ann_info = self.coco.loadAnns(_id)[0]

            d = {
                "id": _id,
                "image": os.path.join(img_dir, img_info["file_name"]),
            }
            if use_mask:
                d.update({"mask": os.path.join(mask_dir, ann_info["file_name"])})
            if use_skeleton:
                skeleton_info = ann_info["skeleton_info"]
                d.update(
                    {"skeleton": os.path.join(skeleton_dir, skeleton_info["file_name"])}
                )
            self.data_list.append(d)

        super().__init__(
            self.data_list,
            transform,
            cache_num=cache_num,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )

    @property
    def num_classes(self):
        return len(self.coco.cats)
