import argparse
import json
import os
from shutil import copyfile

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=str)
    parser.add_argument("output_root_dir", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_name = "imagecas"
    img_dirname = "images"
    mask_dirname = "masks"

    img_dir = os.path.join(args.dataset_dir, img_dirname)
    mask_dir = os.path.join(args.dataset_dir, mask_dirname)

    output_dir = os.path.join(args.output_root_dir, dataset_name)
    assert not os.path.exists(output_dir)
    img_save_dir = os.path.join(output_dir, img_dirname)
    mask_save_dir = os.path.join(output_dir, mask_dirname)
    os.makedirs(img_save_dir)
    os.makedirs(mask_save_dir)

    # https://scikit-learn.org/stable/modules/cross_validation.html
    # test는 별도로 관리해야 맞기 때문에, cross-validation에서 제외하기.
    split_info_path = os.path.join(args.dataset_dir, "imageCAS_data_split.csv")
    split_info_fold_1 = pd.read_csv(split_info_path)[["id", "fold_1"]]
    test_ids = split_info_fold_1[split_info_fold_1["fold_1"] == "test"]["id"].tolist()
    train_val_ids = split_info_fold_1[
        split_info_fold_1["fold_1"].isin(["train", "validation"])
    ]["id"].tolist()

    info = {
        "num_classes": 1,
        "download_url": "https://www.kaggle.com/datasets/xiaoweixumedicalai/imagecas",
        "author": "xiao.wei.xu@foxmail.com",
        "paper_url": "https://arxiv.org/abs/2211.01607",
        "extension": "nii.gz",
    }

    for mode, ids in [("train_val", train_val_ids), ("test", test_ids)]:
        for i in ids:
            img_basename = f"{i}.img.nii.gz"
            mask_basename = f"{i}.label.nii.gz"

            # 다른 label이 추가될 수도 있기 때문에 이름을 동일하게 맞춰준다.
            save_basename = f"{i}.nii.gz"
            img_path = os.path.join(img_dir, img_basename)
            mask_path = os.path.join(mask_dir, mask_basename)
            assert os.path.exists(img_path) and os.path.exists(mask_path)

            copyfile(img_path, os.path.join(img_save_dir, save_basename))
            copyfile(mask_path, os.path.join(mask_save_dir, save_basename))

        # save annotations
        ids_save_path = os.path.join(output_dir, f"{mode}_ids.json")
        with open(ids_save_path, "w") as f:
            json.dump(ids, f)

    info_save_path = os.path.join(output_dir, "dataset.json")
    with open(info_save_path, "w") as f:
        json.dump(info, f)


if __name__ == "__main__":
    main()
