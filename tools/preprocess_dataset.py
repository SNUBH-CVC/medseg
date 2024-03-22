import argparse
import json
import multiprocessing
import os

import numpy as np
from monai.transforms import (Compose, EnsureChannelFirstd, EnsureTyped,
                              LoadImaged, Spacingd, SqueezeDimd)
from sklearn.model_selection import KFold

from medseg.core.utils import setup_logger
from medseg.datasets import ImageCasDataset

logger = setup_logger()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--annotation_filename", type=str, default="train_val.json")
    parser.add_argument("--img_dirname", type=str, default="images")
    parser.add_argument("--mask_dirname", type=str, default="masks")
    return parser.parse_args()


class Preprocessor:
    def __init__(
        self,
        dataset,
        transform,
        output_dir,
        num_processes=None,
    ):
        self.dataset = dataset
        self.transform = transform
        for tr in self.transform.transforms:
            if isinstance(tr, Spacingd):
                self.target_spacing = tr.spacing_transform.pixdim
        else:
            self.target_spacing = None
        self.output_dir = output_dir
        self.num_processes = num_processes
        self.img_save_dir = os.path.join(self.output_dir, "images")
        self.mask_save_dir = os.path.join(self.output_dir, "masks")
        os.makedirs(self.img_save_dir, exist_ok=True)
        os.makedirs(self.mask_save_dir, exist_ok=True)

        self.seed = 42
        self.k = 5
        self.transform = transform
        self.img_key = "image"
        self.mask_key = "mask"

    def split(self):
        kf = KFold(n_splits=self.k, random_state=self.seed, shuffle=True)
        result = []
        for train_indices, val_indices in kf.split(self.dataset):
            result.append(
                {
                    "train": train_indices.tolist(),
                    "val": val_indices.tolist(),
                }
            )
        return result

    def preprocess(self):
        manager = multiprocessing.Manager()
        images = manager.list()
        annotations = manager.list()
        counter = manager.Value("i", 0)
        lock = manager.Lock()
        num_data = len(self.dataset)
        with multiprocessing.Pool(self.num_processes) as pool:
            pool.starmap(
                self._preprocess_single_item,
                [
                    (
                        data,
                        images,
                        annotations,
                        num_data,
                        counter,
                        lock,
                    )
                    for data in self.dataset
                ],
            )

        annotation_data = {
            "info": self.dataset.coco.dataset["info"],
            "images": list(images),
            "annotations": list(annotations),
            "categories": self.dataset.coco.dataset["categories"],
        }
        with open(os.path.join(self.output_dir, "train_val.json"), "w") as f:
            json.dump(annotation_data, f)

        splits = self.split()
        with open(os.path.join(self.output_dir, "splits.json"), "w") as f:
            json.dump(splits, f)

    def _preprocess_single_item(
        self,
        data,
        images,
        annotations,
        num_data,
        counter,
        lock,
    ):
        _id = data["id"]
        img_path = data["image"]
        mask_path = data["mask"]

        # bbox 있는 경우 처리 로직 추가
        data = {self.img_key: img_path, self.mask_key: mask_path}
        with lock:
            counter.value += 1
        logger.info(f"Preprocessing {counter.value}/{num_data}: {img_path}")

        res = self.transform(data)
        img, mask = res[self.img_key], res[self.mask_key]

        basename = f"{_id}.npy"
        images.append(
            {
                "id": _id,
                "file_name": basename,
                "shape": img.shape,
                "spacing": self.target_spacing,
            }
        )
        annotations.append(
            {
                "image_id": _id,
                "mask_info": {
                    "file_name": basename,
                },
            }
        )
        np.save(os.path.join(self.img_save_dir, basename), img)
        np.save(os.path.join(self.mask_save_dir, basename), mask)


def main():
    args = parse_args()
    dataset = ImageCasDataset(
        args.dataset_dir,
        args.annotation_filename,
        args.img_dirname,
        "train_val",
        mask_dirname=args.mask_dirname,
    )
    target_spacing = np.percentile(
        [dataset.coco.load_img(i)["spacing"] for i in dataset.coco.get_img_ids()], 50, 0
    )
    logger.info(f"Adjust target_spacing: {target_spacing}.")
    transforms = Compose(
        [
            LoadImaged(keys=["image", "mask"]),
            EnsureChannelFirstd(keys=["image", "mask"]),
            Spacingd(keys=["image", "mask"], pixdim=target_spacing),
            EnsureTyped(
                keys=["image", "mask"], dtype=[np.float64, np.uint8], data_type="numpy"
            ),
            SqueezeDimd(keys=["image", "mask"], dim=0),
        ]
    )

    preprocessor = Preprocessor(dataset, transforms, args.output_dir)
    preprocessor.preprocess()


if __name__ == "__main__":
    main()
