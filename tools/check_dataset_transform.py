import logging
import multiprocessing
import os

import hydra
import nibabel as nib
import numpy as np
import pyrootutils
from omegaconf import DictConfig

root = pyrootutils.setup_root(
    search_from=__file__,
    pythonpath=True,
)

logger = logging.getLogger(__name__)

HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": str(root / "configs"),
    "config_name": "check_transforms.yaml",
}


def save_nifti_image(img_data, spacing, save_path):
    affine = np.array(
        [
            [spacing[0], 0, 0, 0],
            [0, spacing[1], 0, 0],
            [0, 0, spacing[2], 0],
            [0, 0, 0, 1],
        ]
    )
    img_nifty = nib.Nifti1Image(img_data, affine)
    nib.save(img_nifty, save_path)


def transform_and_save(img, label, spacing, img_save_path, label_save_path):
    save_nifti_image(img, spacing, img_save_path)
    save_nifti_image(label, spacing, label_save_path)


@hydra.main(**HYDRA_PARAMS)
def main(cfg: DictConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)
    num_processes = 4
    train_dataloader = hydra.utils.instantiate(cfg.train_dataloader)
    num_data = len(train_dataloader)

    with multiprocessing.Pool(processes=num_processes) as pool:
        for epoch in range(cfg.num_epochs):
            for i, batch in enumerate(train_dataloader):
                logger.info(f"Running {(i + 1) * len(batch)}/{num_data} items...")
                img_arr = batch["image"].numpy()
                label_arr = batch["label"].numpy()
                spacing_arr = batch["spacing"].numpy()

                pool.starmap(
                    transform_and_save,
                    [
                        (
                            i_arr.squeeze(),
                            l_arr.squeeze(),
                            s_arr.squeeze(),
                            os.path.join(
                                cfg.output_dir,
                                f"img.{epoch:02d}.{i * cfg.batch_size + j:04d}.nii.gz",
                            ),
                            os.path.join(
                                cfg.output_dir,
                                f"label.{epoch:02d}.{i * cfg.batch_size + j:04d}.nii.gz",
                            ),
                        )
                        for j, (i_arr, l_arr, s_arr) in enumerate(
                            zip(img_arr, label_arr, spacing_arr)
                        )
                    ],
                )


if __name__ == "__main__":
    main()
