import json
import os
import sys
import warnings

from monai.data import CacheDataset
from monai.transforms import Randomizable

from .utils import PanopticCOCO


# https://github.com/Project-MONAI/tutorials/blob/main/modules/public_datasets.ipynb
class MpxaDataset(Randomizable, CacheDataset):
    dataset_name = "mpxa"

    def __init__(
        self,
        dataset_dir,
        annotation_filename,
        img_dirname,
        mask_dirname,
        mode,
        transform=None,
        cache_num=sys.maxsize,
        cache_rate=0.0,
        num_workers=0,
        splits_path=None,
        k=1,
    ):
        self.coco = PanopticCOCO(os.path.join(dataset_dir, annotation_filename))
        self.mode = mode
        assert 1 <= k <= 5

        img_dir = os.path.join(dataset_dir, img_dirname)
        mask_dir = os.path.join(dataset_dir, mask_dirname)

        self.data_list = []
        if splits_path is not None:
            with open(os.path.join(dataset_dir, splits_path), "r") as f:
                split_d = json.load(f)
            assert len(split_d) == 5
            self.img_ids = split_d[k - 1][mode]
        else:
            warnings.warn(
                "No `splits_path` has input. It could be for preprocessing or test dataset."
            )
            self.img_ids = self.coco.get_img_ids()

        for _id in self.img_ids:
            img_info = self.coco.load_img(_id)
            ann_info = self.coco.load_ann(_id)

            d = {
                "id": _id,
                "image": os.path.join(img_dir, img_info["file_name"]),
                "mask": os.path.join(mask_dir, ann_info["mask_info"]["file_name"]),
            }
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
