import mlflow
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events


class MedSegSupervisedTrainer:
    def __init__(
        self,
        run_name,
        cfg_path,
        model,
        trainer,
        evaluator,
        train_dataloader,
        val_dataloader,
    ):
        self.run_name = run_name
        self.cfg_path = cfg_path
        self.model = model
        self.trainer = trainer
        self.evaluator = evaluator
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.best_metric = -1

        self.trainer.add_event_handler(Events.STARTED, self.on_started)
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
        self.trainer.add_event_handler(Events.COMPLETED, self.on_completed)

    def on_started(self, engine):
        self.pbar = ProgressBar(persist=True)
        self.pbar.attach(self.trainer, metric_names="all")
        mlflow.start_run(run_name=self.run_name)
        mlflow.log_artifact(self.cfg_path)

    def on_epoch_started(self, engine):
        pass

    def on_epoch_completed(self, engine):
        self.evaluator.run(self.val_dataloader)

    def on_validation_completed(self, engine):
        metrics = self.evaluator.state.metrics
        mean_dice = metrics["mean_dice"]
        mlflow.log_metric(key="mean_dice", value=mean_dice, step=engine.state.epoch)
        if mean_dice > self.best_metric:
            self.best_metric = mean_dice
            mlflow.pytorch.log_model(self.model, "model")
            self.pbar.log_message("saved new best metric model")
        self.pbar.log_message(
            f"Validation Results - Epoch: {engine.state.epoch} Avg accuracy: {mean_dice}"
        )
        self.pbar.n = self.pbar.last_print_n = 0
        pass

    def on_iteration_started(self, engine):
        pass

    def on_iteration_completed(self, engine):
        loss = engine.state.output
        mlflow.log_metric("train_iter_loss", loss, step=engine.state.iteration)

    def on_completed(self, engine):
        mlflow.end_run()

    def run(self, max_epochs):
        self.trainer.run(self.train_dataloader, max_epochs=max_epochs)
