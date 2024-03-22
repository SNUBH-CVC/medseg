import argparse
import json
import multiprocessing
import os
from collections import defaultdict

import numpy as np
from monai.transforms import Compose, EnsureTyped, LoadImaged

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
    return parser.parse_args()


class MetadataCollector:
    def __init__(self, dataset, transform, output_dir: str, num_processes: int = None):
        self.dataset = dataset
        self.transform = transform
        self.output_dir = output_dir
        self.num_processes = num_processes

    def collect_metadata(self):
        os.makedirs(self.output_dir, exist_ok=True)
        manager = multiprocessing.Manager()
        result_list = manager.list()
        num_data = len(self.dataset)
        counter = manager.Value("i", 0)
        lock = manager.Lock()
        with multiprocessing.Pool(processes=self.num_processes) as pool:
            pool.starmap(
                self._get_metadata,
                [(data, result_list, num_data, counter, lock) for data in self.dataset],
            )

        metadata = self._process_metadata(result_list)
        with open(os.path.join(self.output_dir, "dataset_metadata.json"), "w") as f:
            json.dump(metadata, f, cls=NumpyEncoder)
        return metadata

    def _get_metadata(self, data, result_list, num_data, counter, lock):
        with lock:
            counter.value += 1
        logger.info(f"Running {counter.value}/{num_data}: {data['image']}")
        res = self.transform({"image": data["image"], "mask": data["mask"]})
        img_arr = res["image"]
        mask_arr = res["mask"]
        img_on_mask_values_dict = {}
        for i in range(1, self.dataset.num_classes + 1):
            img_on_mask_values_dict.update({i: img_arr[np.where(mask_arr == i)]})

        result = [img_on_mask_values_dict]
        result_list.append(result)

    def _process_metadata(self, result_list):
        metadata = {
            "foreground_intensity_properties_per_channel": {},
            "spacings": [],
        }
        foreground_intensity_properties_per_channel = defaultdict(list)
        for res in result_list:
            for i in range(1, self.dataset.num_classes + 1):
                if i in res[0]:
                    foreground_intensity_properties_per_channel[i].extend(res[0][i])

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
        return metadata


def main():
    args = parse_args()
    transforms = Compose(
        [
            LoadImaged(keys=["image", "mask"]),
            EnsureTyped(
                keys=["image", "mask"], dtype=[np.float64, np.uint8], data_type="numpy"
            ),
        ]
    )
    dataset = ImageCasDataset(
        args.dataset_dir,
        args.annotation_filename,
        args.img_dirname,
        "train_val",
        mask_dirname=args.mask_dirname,
    )

    metadata_collector = MetadataCollector(dataset, transforms, args.output_dir)
    metadata_collector.collect_metadata()


if __name__ == "__main__":
    main()
