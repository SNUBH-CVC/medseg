import importlib.util
import json
import logging
import os

import mlflow
import numpy as np


def import_attribute(module_path, attribute_name):
    # Load the module from the specified path
    spec = importlib.util.spec_from_file_location("module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Return the specified attribute from the module
    return getattr(module, attribute_name)


def get_attributes_from_module(module_path):
    spec = importlib.util.spec_from_file_location("module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Get all variables from the module
    module_variables = {}
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if not attr_name.startswith("__"):
            module_variables[attr_name] = attr

    return module_variables


def set_mlflow_tracking_uri(uri):
    if uri is None:
        uri = os.environ.get("MLFLOW_TRACKING_URI")
    assert uri is not None
    mlflow.set_tracking_uri(uri)


class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def setup_logger(log_path=None):
    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # Set the logging level to DEBUG

    # Create a formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Create a stream handler to log to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Create a file handler to log to a file
    if log_path is not None:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def lambda_lr(max_epochs):
    return lambda epoch: (1 - epoch / max_epochs) ** 0.9


def lambda_prepare_batch(keys):
    return lambda batch, device, non_blocking: (batch[k].to(device) for k in keys)
