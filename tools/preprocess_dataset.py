import argparse
import json
import multiprocessing
import os

import kimimaro
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


def skeletonize(mask: np.uint8):
    anisotropy = (16, 16, 40)
    skels = kimimaro.skeletonize(
        mask,
        teasar_params={
            "scale": 1,
            "const": 200,  # physical units
            "pdrf_scale": 100000,
            "pdrf_exponent": 4,
            "soma_acceptance_threshold": 3500,  # physical units
            "soma_detection_threshold": 750,  # physical units
            "soma_invalidation_const": 300,  # physical units
            "soma_invalidation_scale": 2,
            "max_paths": 300,  # default None
        },
        # object_ids=[ ... ], # process only the specified labels
        # extra_targets_before=[ (27,33,100), (44,45,46) ], # target points in voxels
        # extra_targets_after=[ (27,33,100), (44,45,46) ], # target points in voxels
        dust_threshold=1000,  # skip connected components with fewer than this many voxels
        anisotropy=anisotropy,  # default True
        fix_branching=True,  # default True
        fix_borders=True,  # default True
        fill_holes=False,  # default False
        fix_avocados=False,  # default False
        progress=True,  # default False, show progress bar
        parallel=1,  # <= 0 all cpu, 1 single process, 2+ multiprocess
        parallel_chunk_size=100,  # how many skeletons to process before updating progress bar
    )
    # merge
    board = np.zeros_like(mask)
    for skel in skels.values():
        vertices = skel.vertices
        rescaled_vertices = (vertices / np.array(anisotropy)).astype(int)
        board[
            rescaled_vertices[:, 0], rescaled_vertices[:, 1], rescaled_vertices[:, 2]
        ] = 1
    return board


class Preprocessor:
    def __init__(
        self,
        dataset,
        transform,
        output_dir,
        num_processes=None,
        generate_skeleton=True,
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
        self.generate_skeleton = generate_skeleton

        self.img_save_dir = os.path.join(self.output_dir, "images")
        self.mask_save_dir = os.path.join(self.output_dir, "masks")
        os.makedirs(self.img_save_dir, exist_ok=True)
        os.makedirs(self.mask_save_dir, exist_ok=True)
        if self.generate_skeleton:
            self.skeleton_save_dir = os.path.join(self.output_dir, "skeletons")
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

        # bbox 있는 경우 처리 로직 추가
        data = {IMG_KEY: img_path, MASK_KEY: mask_path}

        res = self.transform(data)
        img, mask = res[IMG_KEY], res[MASK_KEY]

        basename = f"{_id}.npy"
        images.append(
            {
                "id": _id,
                "file_name": basename,
                "shape": img.shape,
                "spacing": self.target_spacing,
            }
        )
        ann = {
            "image_id": _id,
            "mask_info": {
                "file_name": basename,
            },
        }
        if self.generate_skeleton:
            ann.update(
                {
                    "skeleton_info": {
                        "file_name": basename,
                    }
                }
            )
            skeleton = skeletonize(mask)
            np.save(os.path.join(self.skeleton_save_dir, basename), skeleton)
        annotations.append(ann)
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
        skeleton_dirname=args.skeleton_dirname,
    )
    target_spacing = np.percentile(
        [dataset.coco.load_img(i)["spacing"] for i in dataset.coco.get_img_ids()], 50, 0
    )
    logger.info(f"Adjust target_spacing: {target_spacing}.")
    all_keys = [IMG_KEY, MASK_KEY]
    transforms = Compose(
        [
            LoadImaged(keys=all_keys),
            EnsureChannelFirstd(keys=all_keys),
            Spacingd(keys=all_keys, pixdim=target_spacing),
            EnsureTyped(
                keys=all_keys, dtype=[np.float64, np.uint8, np.uint8], data_type="numpy"
            ),
            SqueezeDimd(keys=all_keys, dim=0),
        ]
    )
    preprocessor = Preprocessor(dataset, transforms, args.output_dir)
    preprocessor.preprocess()


if __name__ == "__main__":
    main()
