import argparse
import logging
import types

import torch
import torch.nn as nn
from ignite.engine import (Events, create_supervised_evaluator,
                           create_supervised_trainer)

from medseg.core.config import Config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_path", type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--multi_gpu", action="store_true")
    return parser.parse_args()


def main():
    logger = logging.getLogger(__name__)
    args = parse_args()
    cfg = Config.from_file(args.cfg_path)

    device = args.device
    model = cfg["model"]
    optimizer = cfg["optimizer"]
    if args.multi_gpu and torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)
    max_epochs = cfg["max_epochs"]
    train_dataloader = cfg["train_dataloader"]
    val_dataloader = cfg["val_dataloader"]
    loss_function = cfg["loss_function"]
    trainer_kwargs = cfg["trainer_kwargs"]
    evaluator_kwargs = cfg["evaluator_kwargs"]
    trainer_handlers = cfg["trainer_handlers"]
    evaluator_handlers = cfg["evaluator_handlers"]

    trainer = create_supervised_trainer(
        model, optimizer, loss_function, device=device, **trainer_kwargs
    )
    evaluator = create_supervised_evaluator(model, device=device, **evaluator_kwargs)

    for handler in trainer_handlers:
        handler.attach(trainer)
    for handler in evaluator_handlers:
        if isinstance(handler, types.LambdaType):
            # need trainer as an argument
            handler(trainer).attach(evaluator)
        else:
            handler.attach(evaluator)

    trainer.add_event_handler(
        Events.EPOCH_COMPLETED, lambda _: evaluator.run(val_dataloader)
    )
    trainer.run(train_dataloader, max_epochs)


if __name__ == "__main__":
    main()
