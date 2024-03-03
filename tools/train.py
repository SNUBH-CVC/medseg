import argparse
import importlib.util
import inspect
import os
import sys

# Get the parent directory of the current script (tools/train.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to the Python path
sys.path.append(parent_dir)

parser = argparse.ArgumentParser()
parser.add_argument("cfg_path", type=str)
args = parser.parse_args()

spec = importlib.util.spec_from_file_location("cfg", args.cfg_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
vars = {
    name: value
    for name, value in vars(module).items()
    if not name.startswith("__") and not inspect.ismodule(value)
}

trainer = vars["trainer_cls"](
    run_name=vars["run_name"],
    cfg_path=vars["cfg_path"],
    model=vars["model"],
    trainer=vars["trainer"],
    evaluator=vars["val_evaluator"],
    train_dataloader=vars["train_dataloader"],
    val_dataloader=vars["val_dataloader"],
    logger=vars["logger"],
)
trainer.run()
