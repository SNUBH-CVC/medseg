from .utils import get_attributes_from_module, import_attribute


class Config:

    @staticmethod
    def from_file(cfg_path: str, mode="train"):
        default_cfg_path = "configs/base/default.py"
        dataset_cfg_path = import_attribute(cfg_path, "dataset_cfg_path")
        default_cfg = get_attributes_from_module(default_cfg_path)
        dataset_cfg = get_attributes_from_module(dataset_cfg_path)

        if mode == "train":
            prepare_func = import_attribute(cfg_path, "prepare_train")
        elif mode == "test":
            prepare_func = import_attribute(cfg_path, "prepare_test")
        return prepare_func(**dataset_cfg, **default_cfg)
