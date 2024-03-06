import os
import sys

import pandas as pd
from monai.data import CacheDataset
from monai.transforms import Randomizable


# https://github.com/Project-MONAI/tutorials/blob/main/modules/public_datasets.ipynb
class ImageCasDataset(Randomizable, CacheDataset):
    resource = None
    md5 = None

    def __init__(
        self,
        dataset_dir,
        section,
        transform,
        download=False,
        seed=0,
        cache_num=sys.maxsize,
        cache_rate=1.0,
        num_workers=0,
        k=1,
    ):
        if not os.path.isdir(dataset_dir):
            raise ValueError("Root directory root_dir must be a directory.")
        self.section = section
        self.set_random_state(seed=seed)
        assert 1 <= k <= 4
        split_filename = "imageCAS_data_split.csv"
        split_d = pd.read_csv(os.path.join(dataset_dir, split_filename))[
            ["id", f"fold_{k}"]
        ].values.tolist()
        if download:
            raise ValueError("Download the dataset manually.")

        data = []
        for _id, _section in split_d:
            if _section == section:
                data.append(
                    {
                        "image": os.path.join(
                            dataset_dir, "images", f"{_id}.img.nii.gz"
                        ),
                        "label": os.path.join(
                            dataset_dir, "masks", f"{_id}.label.nii.gz"
                        ),
                    }
                )

        super().__init__(
            data,
            transform,
            cache_num=cache_num,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )
