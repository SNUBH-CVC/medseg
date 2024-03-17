import argparse
import json
import multiprocessing
import os
import sys
from collections import defaultdict

import numpy as np
import SimpleITK as sitk

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medseg.core.utils import NumpyEncoder, import_attribute, setup_logger

logger = setup_logger()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_path", type=str)
    parser.add_argument("output_dir", type=str)
    return parser.parse_args()


def get_metadata(
    img_path, mask_path, num_classes, result_list, num_data, counter, lock
):
    with lock:
        counter.value += 1
    logger.info(f"Running {counter.value}/{num_data}: {img_path}")
    img_on_mask_values_dict = {}
    img = sitk.ReadImage(img_path)
    spacing = img.GetSpacing()
    mask_arr = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
    img_arr = sitk.GetArrayFromImage(img)
    for i in range(1, num_classes + 1):
        img_on_mask_values_dict.update({i: img_arr[np.where(mask_arr == i)]})

    result = [spacing, img_on_mask_values_dict]
    result_list.append(result)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train_dataset = import_attribute(args.cfg_path, "train_dataset")
    img_path_list = []
    mask_path_list = []
    for d in train_dataset.data_list:
        img_path_list.append(d["image"])
        mask_path_list.append(d["label"])

    manager = multiprocessing.Manager()
    result_list = manager.list()
    num_data = len(img_path_list)
    counter = manager.Value("i", 0)
    lock = manager.Lock()
    num_classes = train_dataset.num_classes
    num_processes = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.starmap(
            get_metadata,
            [
                (img_path, mask_path, num_classes, result_list, num_data, counter, lock)
                for i, (img_path, mask_path) in enumerate(
                    zip(img_path_list, mask_path_list)
                )
            ],
        )

    meta_data = {
        "foreground_intensity_properties_per_channel": {},
        "spacings": [],
    }
    foreground_intensity_properties_per_channel = defaultdict(list)
    spacing_list = []
    for res in result_list:
        spacing_list.append(res[0])
        for i in range(1, num_classes + 1):
            if i in res[1]:
                foreground_intensity_properties_per_channel[i].extend(res[1][i])

    for k, v in foreground_intensity_properties_per_channel.items():
        meta_data["foreground_intensity_properties_per_channel"].update(
            {
                k: {
                    "mean": np.mean(v),
                    "median": np.median(v),
                    "min": np.min(v),
                    "max": np.max(v),
                    "percentile_00_5": np.percentile(v, 0.005),
                    "percentile_99_5": np.percentile(v, 0.995),
                    "std": np.std(v),
                }
            }
        )
    meta_data.update({"spacings": spacing_list})
    with open(os.path.join(args.output_dir, "dataset_metadata.json"), "w") as f:
        json.dump(meta_data, f, cls=NumpyEncoder)
    return meta_data


if __name__ == "__main__":
    main()
