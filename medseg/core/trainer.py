import logging
import os

import mlflow
from ignite.engine import (Events, create_supervised_evaluator,
                           create_supervised_trainer)


class SupervisedTrainWrapper:
    def __init__(
        self,
        cfg_path,
        device,
        model,
        optimizer,
        loss_function,
        key_metric,
        max_epochs,
        train_dataloader=None,
        val_dataloader=None,
        trainer_kwargs=None,
        evaluator_kwargs=None,
        logger=None,
    ):
        self.cfg_path = cfg_path
        self.run_name = os.path.splitext(os.path.basename(cfg_path))[0]
        self.device = device
        self.model = model.to(device)
        self.key_metric = key_metric
        self.max_epochs = max_epochs
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.trainer = create_supervised_trainer(
            model, optimizer, loss_function, device=device, **trainer_kwargs
        )
        self.evaluator = create_supervised_evaluator(
            model, device=device, **evaluator_kwargs
        )
        self.best_metric = -1

        self.trainer.add_event_handler(Events.STARTED, self.on_train_started)
        self.trainer.add_event_handler(Events.EPOCH_STARTED, self.on_epoch_started)
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self.on_epoch_completed)
        self.evaluator.add_event_handler(
            Events.EPOCH_COMPLETED, self.on_validation_completed
        )
        self.trainer.add_event_handler(
            Events.ITERATION_STARTED, self.on_iteration_started
        )
        self.trainer.add_event_handler(
            Events.ITERATION_COMPLETED, self.on_iteration_completed
        )
        self.trainer.add_event_handler(Events.COMPLETED, self.on_train_completed)
        if logger is None:
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger()
        self.logger = logger

    def on_train_started(self, engine):
        mlflow.start_run(run_name=self.run_name)
        mlflow.log_artifact(self.cfg_path, "config")

    def on_epoch_started(self, engine):
        pass

    def on_epoch_completed(self, engine):
        self.evaluator.run(self.val_dataloader)

    def on_validation_completed(self, engine):
        metrics = self.evaluator.state.metrics
        key_metric = metrics[self.key_metric]
        mlflow.log_metric(
            key=self.key_metric, value=key_metric, step=engine.state.epoch
        )
        if key_metric > self.best_metric:
            self.best_metric = key_metric
            mlflow.pytorch.log_model(self.model, "model")
            self.logger.info("saved new best metric model")
        self.logger.info(
            f"Validation results - epoch: {engine.state.epoch} {self.key_metric}: {key_metric}"
        )

    def on_iteration_started(self, engine):
        pass

    def on_iteration_completed(self, engine):
        loss = engine.state.output
        mlflow.log_metric("train_iter_loss", loss, step=engine.state.iteration)

    def on_train_completed(self, engine):
        mlflow.end_run()

    def run(self):
        self.trainer.run(self.train_dataloader, max_epochs=self.max_epochs)
