import importlib.util
import os

import mlflow


def import_attribute(module_path, attribute_name):
    # Load the module from the specified path
    spec = importlib.util.spec_from_file_location("module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Return the specified attribute from the module
    return getattr(module, attribute_name)


def set_mlflow_tracking_uri(uri):
    if uri is None:
        uri = os.environ.get("MLFLOW_TRACKING_URI")
    assert uri is not None
    mlflow.set_tracking_uri(uri)
