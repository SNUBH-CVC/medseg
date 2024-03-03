import logging
from abc import ABC, abstractmethod

import mlflow
from ignite.engine import Events


# test는 분리해도 될 듯
class BaseSupervisedTrainer(ABC):
    def __init__(
        self,
        run_name=None,
        cfg_path=None,
        model=None,
        trainer=None,
        evaluator=None,
        train_dataloader=None,
        val_dataloader=None,
        logger=None,
    ):
        self.run_name = run_name
        self.cfg_path = cfg_path
        self.model = model
        self.trainer = trainer
        self.evaluator = evaluator
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
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
        mlflow.log_artifact(self.cfg_path)

    @abstractmethod
    def on_epoch_started(self, engine):
        pass

    def on_epoch_completed(self, engine):
        self.evaluator.run(self.val_dataloader)

    @abstractmethod
    def on_validation_completed(self, engine):
        pass

    @abstractmethod
    def on_iteration_started(self, engine):
        pass

    @abstractmethod
    def on_iteration_completed(self, engine):
        pass

    def on_train_completed(self, engine):
        mlflow.end_run()

    def run(self, max_epochs):
        self.trainer.run(self.train_dataloader, max_epochs=max_epochs)


class ImageCasSupervisedTrainer(BaseSupervisedTrainer):

    def on_epoch_started(self, engine):
        pass

    def on_validation_completed(self, engine):
        metrics = self.evaluator.state.metrics
        mean_dice = metrics["mean_dice"]
        mlflow.log_metric(key="mean_dice", value=mean_dice, step=engine.state.epoch)
        if mean_dice > self.best_metric:
            self.best_metric = mean_dice
            mlflow.pytorch.log_model(self.model, "model")
            self.logger.info("saved new best metric model")
        self.logger.info(
            f"Validation results - Epoch: {engine.state.epoch} Avg accuracy: {mean_dice}"
        )

    def on_iteration_started(self, engine):
        pass

    def on_iteration_completed(self, engine):
        loss = engine.state.output
        mlflow.log_metric("train_iter_loss", loss, step=engine.state.iteration)


class BaseSupervisedTester:

    def __init__(self, dataloader, evaluator):
        self.dataloader = dataloader
        self.evaluator = evaluator

    @abstractmethod
    def on_test_iteration_completed(self, engine):
        pass

    @abstractmethod
    def on_test_completed(self, engine):
        pass

    def test(self):
        self.evaluator.run(self.dataloader)


class ImageCasSupervisedTester(BaseSupervisedTester):
    pass
