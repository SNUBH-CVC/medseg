import glob
import os
import sys

from monai.data import CacheDataset
from monai.transforms import Randomizable


# https://github.com/Project-MONAI/tutorials/blob/main/modules/public_datasets.ipynb
class ImageCASDataset(Randomizable, CacheDataset):
    resource = None
    md5 = None

    def __init__(
        self,
        root_dir,
        section,
        transform,
        download=False,
        seed=0,
        val_frac=0.2,
        test_frac=0.2,
        cache_num=sys.maxsize,
        cache_rate=1.0,
        num_workers=0,
    ):
        if not os.path.isdir(root_dir):
            raise ValueError("Root directory root_dir must be a directory.")
        self.section = section
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.set_random_state(seed=seed)
        dataset_dir = os.path.join(root_dir, "imagecas")
        # split_filename = "imageCAS_data_split.xlsx"
        if download:
            raise ValueError("Download the dataset manually.")

        img_dir = os.path.join(dataset_dir, "images")
        img_list = glob.glob(os.path.join(img_dir, "*.img.nii.gz"))
        self.datalist = []
        for img_path in img_list:
            img_basename = os.path.basename(img_path)
            label_basename = img_basename.replace("img", "label")
            self.datalist.append(
                {"image": img_path, "label": os.path.join(img_dir, label_basename)}
            )

        data = self._generate_data_list()
        super().__init__(
            data,
            transform,
            cache_num=cache_num,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )

    def randomize(self, data=None):
        self.rann = self.R.random()

    def _generate_data_list(self):
        data = []
        for d in self.datalist:
            self.randomize()
            if self.section == "training":
                if self.rann < self.val_frac + self.test_frac:
                    continue
            elif self.section == "validation":
                if self.rann >= self.val_frac:
                    continue
            elif self.section == "test":
                if (
                    self.rann < self.val_frac
                    or self.rann >= self.val_frac + self.test_frac
                ):
                    continue
            else:
                raise ValueError(
                    f"Unsupported section: {self.section}, "
                    "available options are ['training', 'validation', 'test']."
                )
            data.append(d)
        return data
