import argparse
import glob
import json
import multiprocessing
import os

import numpy as np
import pydicom
import tqdm
from skimage.draw import polygon2mask
from sklearn.model_selection import KFold

from medseg.core.utils import setup_logger

logger = setup_logger()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=str)
    parser.add_argument("output_dir", type=str)
    return parser.parse_args()


def normalize(pixel_array, window_center, window_width):
    # Calculate the min and max values for the window
    window_min = window_center - window_width / 2
    window_max = window_center + window_width / 2

    # Apply windowing
    windowed_array = np.clip(pixel_array, window_min, window_max)

    # Scale the pixel values to [0, 1]
    scaled_array = (windowed_array - window_min) / (window_max - window_min)
    return scaled_array


def run_single(dcm_path, result_path, img_save_dir, mask_save_dir, images, annotations):
    dcm = pydicom.dcmread(dcm_path)
    with open(result_path, "r") as f:
        data = json.load(f)
    frame_number = data["frameNo"]
    img = dcm.pixel_array[frame_number]
    img = normalize(img, dcm.WindowCenter, dcm.WindowWidth)
    contour = np.array(json.loads(data["editContour"]))
    mask = polygon2mask(img.shape[:2], contour[:, ::-1]).astype(np.uint8)
    token = data["token"]
    basename = f"{token}.npy"

    images.append(
        {
            "id": token,
            "file_name": basename,
            "shape": img.shape,
        }
    )
    annotations.append({"image_id": token, "mask_info": {"file_name": basename}})
    np.save(os.path.join(img_save_dir, basename), img)
    np.save(os.path.join(mask_save_dir, basename), mask)


def main():
    args = parse_args()
    assert not os.path.exists(args.output_dir)

    result_path_list = glob.glob(
        os.path.join(args.dataset_dir, "**/*.json"), recursive=True
    )
    dcm_path_list = [i.replace("json", "dcm") for i in result_path_list]
    token_list = [os.path.splitext(os.path.basename(i))[0] for i in result_path_list]
    img_save_dir = os.path.join(args.output_dir, "images")
    mask_save_dir = os.path.join(args.output_dir, "masks")
    os.makedirs(img_save_dir)
    os.makedirs(mask_save_dir)

    info = {
        "description": "SNUBH-CVC MPXA dataset",
        "author": "whikwon@gmail.com",
    }
    categories = [
        {
            "supercategory": "medical",
            "id": 1,
            "name": "coronary",
            "color": 1,
        }
    ]
    splits = []
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    for train_indices, val_indices in kf.split(token_list):
        train_ids = [token_list[i] for i in train_indices]
        val_ids = [token_list[i] for i in val_indices]
        splits.append(
            {
                "train": train_ids,
                "val": val_ids,
            }
        )
    with open(os.path.join(args.output_dir, "splits.json"), "w") as f:
        json.dump(splits, f)

    manager = multiprocessing.Manager()
    images = manager.list()
    annotations = manager.list()
    with multiprocessing.Pool() as pool:
        pool.starmap(
            run_single,
            tqdm.tqdm(
                [
                    (
                        dcm_path,
                        result_path,
                        img_save_dir,
                        mask_save_dir,
                        images,
                        annotations,
                    )
                    for dcm_path, result_path in zip(dcm_path_list, result_path_list)
                ],
                total=len(dcm_path_list),
            ),
        )

        annotation_data = {
            "info": info,
            "categories": categories,
            "images": list(images),
            "annotations": list(annotations),
        }
        annotation_save_path = os.path.join(args.output_dir, f"train_val.json")
        with open(annotation_save_path, "w") as f:
            json.dump(annotation_data, f)


if __name__ == "__main__":
    main()
