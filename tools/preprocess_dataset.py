import argparse
import json
import multiprocessing
import os
from collections import defaultdict

import numpy as np
import SimpleITK as sitk
from monai.transforms import Compose, EnsureTyped, LoadImaged, Spacingd
from sklearn.model_selection import KFold

from medseg.core.utils import NumpyEncoder, setup_logger

logger = setup_logger()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=str)
    parser.add_argument("output_dir", type=str)
    return parser.parse_args()


def get_train_val_dataset(dataset_dir):
    with open(os.path.join(dataset_dir, "dataset.json"), "r") as f:
        info = json.load(f)
    with open(os.path.join(dataset_dir, "train_val_ids.json"), "r") as f:
        ids = json.load(f)
    extension = info["extension"]

    dataset = []
    for _id in ids:
        basename = f"{_id}.{extension}"
        dataset.append(
            [
                _id,
                os.path.join(dataset_dir, "images", basename),
                os.path.join(dataset_dir, "masks", basename),
            ]
        )
    return dataset, info["num_classes"]


class MetadataCollector:
    def __init__(self, dataset, num_classes, output_dir):
        self.dataset = dataset
        self.num_classes = num_classes
        self.output_dir = output_dir

    def collect_fingerprint(self):
        os.makedirs(self.output_dir)
        manager = multiprocessing.Manager()
        result_list = manager.list()
        num_data = len(self.dataset)
        counter = manager.Value("i", 0)
        lock = manager.Lock()
        num_processes = multiprocessing.cpu_count()
        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.starmap(
                self._get_metadata,
                [
                    (img_path, mask_path, result_list, num_data, counter, lock)
                    for _id, img_path, mask_path in self.dataset
                ],
            )

        metadata = self._process_metadata(result_list)
        with open(os.path.join(self.output_dir, "dataset_fingerprint.json"), "w") as f:
            json.dump(metadata, f, cls=NumpyEncoder)
        return metadata

    def _get_metadata(self, img_path, mask_path, result_list, num_data, counter, lock):
        with lock:
            counter.value += 1
        logger.info(f"Running {counter.value}/{num_data}: {img_path}")
        img_on_mask_values_dict = {}
        img = sitk.ReadImage(img_path)
        spacing = img.GetSpacing()
        mask_arr = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
        img_arr = sitk.GetArrayFromImage(img)
        for i in range(1, self.num_classes + 1):
            img_on_mask_values_dict.update({i: img_arr[np.where(mask_arr == i)]})

        result = [spacing, img_on_mask_values_dict]
        result_list.append(result)

    def _process_metadata(self, result_list):
        metadata = {
            "foreground_intensity_properties_per_channel": {},
            "spacings": [],
        }
        foreground_intensity_properties_per_channel = defaultdict(list)
        spacing_list = []
        for res in result_list:
            spacing_list.append(res[0])
            for i in range(1, self.num_classes + 1):
                if i in res[1]:
                    foreground_intensity_properties_per_channel[i].extend(res[1][i])

        for k, v in foreground_intensity_properties_per_channel.items():
            metadata["foreground_intensity_properties_per_channel"].update(
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
        metadata.update({"spacings": spacing_list})
        return metadata


class Preprocessor:
    def __init__(self, dataset, target_spacing, output_dir):
        self.dataset = dataset
        self.target_spacing = target_spacing
        self.output_dir = output_dir
        self.seed = 42
        self.k = 5

    def split(self):
        kf = KFold(n_splits=self.k, random_state=self.seed)
        result = []
        for train_indices, val_indices in kf.split(self.dataset):
            result.append(
                {
                    "train": [self.dataset[i] for i in train_indices],
                    "val": [self.dataset[i] for i in val_indices],
                }
            )
        return result

    def preprocess(self):
        num_data = len(self.dataset)
        manager = multiprocessing.Manager()
        result_list = manager.list()
        counter = manager.Value("i", 0)
        lock = manager.Lock()
        img_save_dir = os.path.join(self.output_dir, "images")
        mask_save_dir = os.path.join(self.output_dir, "masks")
        os.makedirs(img_save_dir)
        os.makedirs(mask_save_dir)
        with multiprocessing.Pool() as pool:
            pool.starmap(
                self._preprocess_single_item,
                [
                    (
                        _id,
                        img_path,
                        mask_path,
                        result_list,
                        img_save_dir,
                        mask_save_dir,
                        num_data,
                        counter,
                        lock,
                    )
                    for _id, img_path, mask_path in self.dataset
                ],
            )
        metadata = {}
        shapes = [i[0] for i in result_list]
        metadata.update(
            {
                "spacing": self.target_spacing,
                "shapes": shapes,
                "median_shape": np.percentile(shapes, 50, 0),
            }
        )
        with open(os.path.join(self.output_dir, "preprocess.json"), "w") as f:
            json.dump(list(result_list), f)

        splits = self.split()
        with open(os.path.join(self.output_dir, "splits.json"), "w") as f:
            json.dump(splits, f)

    def _preprocess_single_item(
        self,
        _id,
        img_path,
        mask_path,
        result_list,
        img_save_dir,
        mask_save_dir,
        num_data,
        counter,
        lock,
    ):
        with lock:
            counter.value += 1
        logger.info(f"Preprocessing {counter.value}/{num_data}: {img_path}")

        keys = ["image", "mask"]
        compose = Compose(
            [
                LoadImaged(keys=keys),
                Spacingd(keys=keys, pixdim=self.target_spacing),
                EnsureTyped(keys=keys, dtype=[np.float64, np.uint8], data_type="numpy"),
            ]
        )
        result = compose({"image": img_path, "mask": mask_path})
        img, mask = result["image"], result["mask"]

        basename = f"{_id}.npy"
        np.save(os.path.join(img_save_dir, basename), img)
        np.save(os.path.join(mask_save_dir, basename), mask)
        result_list.append([img.shape])


def main():
    args = parse_args()
    dataset, num_classes = get_train_val_dataset(args.dataset_dir)

    # Collect metadata
    metadata_collector = MetadataCollector(dataset, num_classes, args.output_dir)
    fingerprint = metadata_collector.collect_fingerprint()

    # Calculate target spacing
    target_spacing = np.percentile(fingerprint["spacings"], 50, 0)

    # Preprocess data
    preprocessor = Preprocessor(dataset, target_spacing, args.output_dir)
    preprocessor.preprocess()


if __name__ == "__main__":
    main()
