import argparse
import json
import multiprocessing
import os
from shutil import copyfile

import nibabel as nib
import pandas as pd

from medseg.core.utils import NumpyEncoder, setup_logger

logger = setup_logger()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=str)
    parser.add_argument("output_root_dir", type=str)
    return parser.parse_args()


def run_single(
    img_id,
    img_path,
    mask_path,
    img_save_dir,
    mask_save_dir,
    images,
    annotations,
    num_data,
    counter,
    lock,
):
    with lock:
        counter.value += 1
    logger.info(f"Running {counter.value}/{num_data}: {img_path}")
    save_basename = f"{img_id}.nii.gz"
    img = nib.load(img_path)
    spacing = img.header.get_zooms()
    img_arr = img.get_fdata()
    assert os.path.exists(img_path) and os.path.exists(mask_path)

    images.append(
        {
            "id": img_id,
            "file_name": save_basename,
            "shape": img_arr.shape,
            "spacing": spacing,
        }
    )
    annotations.append(
        {
            "image_id": img_id,
            "mask_info": {
                "file_name": save_basename,
            },
        }
    )
    copyfile(img_path, os.path.join(img_save_dir, save_basename))
    copyfile(mask_path, os.path.join(mask_save_dir, save_basename))


def main():
    args = parse_args()
    dataset_name = "imagecas"
    img_dir = os.path.join(args.dataset_dir, "images")
    mask_dir = os.path.join(args.dataset_dir, "masks")

    # dataset 이름으로 자동 지정
    output_dir = os.path.join(args.output_root_dir, dataset_name)
    assert not os.path.exists(output_dir)
    img_save_dir = os.path.join(output_dir, "images")
    mask_save_dir = os.path.join(output_dir, "masks")
    os.makedirs(img_save_dir)
    os.makedirs(mask_save_dir)

    # https://scikit-learn.org/stable/modules/cross_validation.html
    # test는 별도로 관리해야 맞기 때문에, cross-validation에서 제외하기.
    split_info_path = os.path.join(args.dataset_dir, "imageCAS_data_split.csv")
    split_info_fold_1 = pd.read_csv(split_info_path)[["id", "fold_1"]]
    test_ids = (
        split_info_fold_1[split_info_fold_1["fold_1"] == "test"]["id"]
        .astype("string")
        .tolist()
    )
    train_val_ids = (
        split_info_fold_1[split_info_fold_1["fold_1"].isin(["train", "validation"])][
            "id"
        ]
        .astype("string")
        .tolist()
    )

    # dataset format: https://github.com/cocodataset/panopticapi
    for mode, ids in [("train_val", train_val_ids), ("test", test_ids)]:
        info = {
            "description": "A-Large-Scale-Dataset-and-Benchmark-for-Coronary-Artery-Segmentation-based-on-CT",
            "download_url": "https://www.kaggle.com/datasets/xiaoweixumedicalai/imagecas",
            "author": "xiao.wei.xu@foxmail.com",
            "paper_url": "https://arxiv.org/abs/2211.01607",
        }
        categories = [
            {
                "supercategory": "medical",
                "id": 1,
                "name": "coronary",
                "color": 1,
            }
        ]
        manager = multiprocessing.Manager()
        images = manager.list()
        annotations = manager.list()
        num_ids = len(ids)
        counter = manager.Value("i", 0)
        lock = manager.Lock()
        with multiprocessing.Pool() as pool:
            pool.starmap(
                run_single,
                [
                    (
                        i,
                        os.path.join(img_dir, f"{i}.img.nii.gz"),
                        os.path.join(mask_dir, f"{i}.label.nii.gz"),
                        img_save_dir,
                        mask_save_dir,
                        images,
                        annotations,
                        num_ids,
                        counter,
                        lock,
                    )
                    for i in ids
                ],
            )

        # save annotations
        annotation_data = {
            "info": info,
            "categories": categories,
            "images": list(images),
            "annotations": list(annotations),
        }
        annotation_save_path = os.path.join(output_dir, f"{mode}.json")
        with open(annotation_save_path, "w") as f:
            json.dump(annotation_data, f, cls=NumpyEncoder)


if __name__ == "__main__":
    main()
