import argparse
import json
import multiprocessing
import os

import numpy as np
import tqdm
from monai.transforms import (Compose, EnsureChannelFirstd, EnsureTyped,
                              LoadImaged, Spacingd, SqueezeDimd)
from sklearn.model_selection import KFold

from medseg.core.utils import NumpyEncoder, setup_logger
from medseg.datasets import ImageCasDataset

logger = setup_logger()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--annotation_filename", type=str, default="train_val.json")
    parser.add_argument("--img_dirname", type=str, default="images")
    parser.add_argument("--mask_dirname", type=str, default="masks")
    parser.add_argument("--skeleton_dirname", type=str, default="skeletons")
    return parser.parse_args()


IMG_KEY = "image"
MASK_KEY = "mask"
SKELETON_KEY = "skeleton"


class Preprocessor:
    def __init__(
        self,
        dataset,
        transform,
        output_dir,
        img_dirname,
        mask_dirname,
        skeleton_dirname,
        num_processes=None,
    ):
        self.dataset = dataset
        self.transform = transform
        for tr in self.transform.transforms:
            if isinstance(tr, Spacingd):
                self.target_spacing = tr.spacing_transform.pixdim
                break
        else:
            self.target_spacing = None
        self.output_dir = output_dir
        self.num_processes = num_processes
        self.img_save_dir = os.path.join(self.output_dir, img_dirname)
        self.mask_save_dir = os.path.join(self.output_dir, mask_dirname)
        self.skeleton_save_dir = os.path.join(self.output_dir, skeleton_dirname)
        os.makedirs(self.img_save_dir, exist_ok=True)
        os.makedirs(self.mask_save_dir, exist_ok=True)
        os.makedirs(self.skeleton_save_dir, exist_ok=True)

        self.seed = 42
        self.k = 5
        self.transform = transform

    def split(self):
        kf = KFold(n_splits=self.k, random_state=self.seed, shuffle=True)
        result = []
        img_ids = self.dataset.coco.get_img_ids()
        for train_indices, val_indices in kf.split(self.dataset):
            train_ids = [img_ids[i] for i in train_indices]
            val_ids = [img_ids[i] for i in val_indices]
            result.append(
                {
                    "train": train_ids,
                    "val": val_ids,
                }
            )
        return result

    def preprocess(self):
        manager = multiprocessing.Manager()
        images = manager.list()
        annotations = manager.list()
        with multiprocessing.Pool(self.num_processes) as pool:
            pool.starmap(
                self._preprocess_single_item,
                tqdm.tqdm(
                    [
                        (
                            data,
                            images,
                            annotations,
                        )
                        for data in self.dataset
                    ],
                    total=len(self.dataset),
                ),
            )

        annotation_data = {
            "info": self.dataset.coco.dataset["info"],
            "images": list(images),
            "annotations": list(annotations),
            "categories": self.dataset.coco.dataset["categories"],
        }
        with open(os.path.join(self.output_dir, "train_val.json"), "w") as f:
            json.dump(annotation_data, f, cls=NumpyEncoder)

        splits = self.split()
        with open(os.path.join(self.output_dir, "splits.json"), "w") as f:
            json.dump(splits, f)

    def _preprocess_single_item(
        self,
        data,
        images,
        annotations,
    ):
        _id = data["id"]
        img_path = data["image"]
        mask_path = data["mask"]
        skeleton_path = data["skeleton"]

        # bbox 있는 경우 처리 로직 추가
        data = {IMG_KEY: img_path, MASK_KEY: mask_path, SKELETON_KEY: skeleton_path}

        res = self.transform(data)
        img, mask, skeleton = res[IMG_KEY], res[MASK_KEY], res[SKELETON_KEY]

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
                "skeleton_info": {
                    "file_name": basename,
                },
            }
        )
        np.save(os.path.join(self.img_save_dir, basename), img)
        np.save(os.path.join(self.mask_save_dir, basename), mask)
        np.save(os.path.join(self.skeleton_save_dir, basename), skeleton)


def main():
    args = parse_args()
    dataset = ImageCasDataset(
        args.dataset_dir,
        args.annotation_filename,
        args.img_dirname,
        "train_val",
        mask_dirname=args.mask_dirname,
        skeleton_dirname=args.skeleton_dirname,
    )
    target_spacing = np.percentile(
        [dataset.coco.load_img(i)["spacing"] for i in dataset.coco.get_img_ids()], 50, 0
    )
    logger.info(f"Adjust target_spacing: {target_spacing}.")

    all_keys = [IMG_KEY, MASK_KEY, SKELETON_KEY]
    transforms = Compose(
        [
            LoadImaged(keys=all_keys),
            EnsureChannelFirstd(keys=all_keys),
            Spacingd(
                keys=all_keys,
                pixdim=target_spacing,
                mode=["bilinear", "nearest", "nearest"],
            ),
            EnsureTyped(
                keys=all_keys,
                dtype=[np.float64, np.uint8, np.uint8],
                data_type="numpy",
            ),
            SqueezeDimd(keys=all_keys, dim=0),
        ]
    )

    preprocessor = Preprocessor(
        dataset,
        transforms,
        args.output_dir,
        args.img_dirname,
        args.mask_dirname,
        args.skeleton_dirname,
    )
    preprocessor.preprocess()


if __name__ == "__main__":
    main()
