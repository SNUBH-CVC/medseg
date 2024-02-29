import os


def get_basename_wo_ext(file_path: str):
    basename = os.path.basename(file_path)
    return ".".join(basename.split(".")[:-1])
