import argparse
import multiprocessing
import os
import sys

import nibabel as nib
import numpy as np
from monai.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medseg.core.utils import get_attributes_from_module, setup_logger

logger = setup_logger()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_path", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--num_epochs", type=int, default=1)
    return parser.parse_args()


def save_nifti(img_data, save_path):
    img_nifty = nib.Nifti1Image(img_data, np.eye(4))
    nib.save(img_nifty, save_path)


def transform_and_save(img, label, img_save_path, label_save_path):
    save_nifti(img, img_save_path)
    save_nifti(label, label_save_path)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    attrs = get_attributes_from_module(args.cfg_path)
    train_dataset = attrs["train_dataset"]
    num_processes = 4
    num_data = len(train_dataset) * args.num_epochs
    batch_size = num_processes
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_processes
    )

    with multiprocessing.Pool(processes=num_processes) as pool:
        for epoch in range(args.num_epochs):
            for i, batch in enumerate(train_dataloader):
                logger.info(f"Running {(i + 1) * batch_size}/{num_data} items...")
                img_arr = batch["image"].numpy()
                label_arr = batch["label"].numpy()

                pool.starmap(
                    transform_and_save,
                    [
                        (
                            i_arr.squeeze(),
                            l_arr.squeeze(),
                            os.path.join(
                                args.output_dir,
                                f"img.{epoch:02d}.{i * batch_size + j:04d}.nii.gz",
                            ),
                            os.path.join(
                                args.output_dir,
                                f"label.{epoch:02d}.{i * batch_size + j:04d}.nii.gz",
                            ),
                        )
                        for j, (i_arr, l_arr) in enumerate(zip(img_arr, label_arr))
                    ],
                )


if __name__ == "__main__":
    main()
