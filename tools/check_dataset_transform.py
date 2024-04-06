import logging
import multiprocessing
import os

import hydra
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


def save_images(img, label, idx, img_saver, label_saver):
    img.meta["patch_index"] = idx
    label.meta["patch_index"] = idx
    img_saver(img)
    label_saver(label)


@hydra.main(**HYDRA_PARAMS)
def main(cfg: DictConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)
    num_processes = 4
    train_dataloader = hydra.utils.instantiate(cfg.dataset.train_dataloader)
    num_data = len(train_dataloader)
    img_saver = hydra.utils.instantiate(cfg.image_saver)
    label_saver = hydra.utils.instantiate(cfg.label_saver)

    with multiprocessing.Pool(processes=num_processes) as pool:
        for epoch in range(cfg.num_epochs):
            for i, batch in enumerate(train_dataloader):
                img = batch["image"]
                label = batch["mask"]
                logger.info(f"Running {(i + 1) * len(img)}/{num_data} items...")

                pool.starmap(
                    save_images,
                    [
                        (
                            i_arr[0],
                            l_arr[0],
                            epoch,
                            img_saver,
                            label_saver,
                        )
                        for i_arr, l_arr in zip(img, label)
                    ],
                )


if __name__ == "__main__":
    main()
