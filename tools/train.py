import argparse
import logging
import os
import sys

import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ignite.engine import (Events, create_supervised_evaluator,
                           create_supervised_trainer)

from medseg.core.utils import import_attribute


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_path", type=str)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--multi_gpu", action="store_true")
    return parser.parse_args()


def main():
    logger = logging.getLogger(__name__)
    args = parse_args()
    prepare_train_func = import_attribute(args.cfg_path, "prepare_train")
    items = prepare_train_func()

    device = args.device
    model = items["model"]
    optimizer = items["optimizer"]
    if args.multi_gpu and torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)
    max_epochs = items["max_epochs"]
    train_dataloader = items["train_dataloader"]
    val_dataloader = items["val_dataloader"]
    loss_function = items["loss_function"]
    trainer_kwargs = items["trainer_kwargs"]
    evaluator_kwargs = items["evaluator_kwargs"]
    trainer_handlers = items["trainer_handlers"]
    evaluator_handlers = items["evaluator_handlers"]

    trainer = create_supervised_trainer(
        model, optimizer, loss_function, device=device, **trainer_kwargs
    )
    evaluator = create_supervised_evaluator(model, device=device, **evaluator_kwargs)

    for handler in trainer_handlers:
        handler.attach(trainer)
    for handler in evaluator_handlers:
        handler.attach(evaluator)

    trainer.add_event_handler(
        Events.EPOCH_COMPLETED, lambda _: evaluator.run(val_dataloader)
    )
    trainer.run(train_dataloader, max_epochs)


if __name__ == "__main__":
    main()
