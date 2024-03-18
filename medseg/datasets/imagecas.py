import json
import os
import sys

from monai.data import CacheDataset
from monai.transforms import Randomizable


# https://github.com/Project-MONAI/tutorials/blob/main/modules/public_datasets.ipynb
class ImageCasDataset(Randomizable, CacheDataset):
    dataset_name = "imagecas"
    num_classes = 1

    def __init__(
        self,
        dataset_dir,
        splits_path,
        mode,
        transform,
        download=False,
        cache_num=sys.maxsize,
        cache_rate=1.0,
        num_workers=0,
        use_mask=True,
        use_skeleton=False,
        k=1,
    ):
        if not os.path.isdir(dataset_dir):
            raise ValueError("Root directory root_dir must be a directory.")
        self.mode = mode
        assert 1 <= k <= 4
        if download:
            raise ValueError("Download the dataset manually.")

        assert use_mask or use_skeleton
        self.data_list = []
        with open(splits_path, "r") as f:
            split_d = json.load(splits_path)
        assert len(split_d) == 5

        for _id, _section in split_d[k]:
            if _section == mode:
                d = {
                    "image": os.path.join(dataset_dir, "images", f"{_id}.npy"),
                }
                if use_mask:
                    d.update({"mask": os.path.join(dataset_dir, "masks", f"{_id}.npy")})
                if use_skeleton:
                    d.update(
                        {
                            "skeleton": os.path.join(
                                dataset_dir, "skeletons", f"{_id}.npy"
                            )
                        }
                    )
                self.data_list.append(d)

        super().__init__(
            self.data_list,
            transform,
            cache_num=cache_num,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )
