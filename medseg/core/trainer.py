import logging

import mlflow
from monai.data import decollate_batch


class Trainer:
    def __init__(
        self,
        run_name,
        device,
        model,
        train_dataloader,
        val_dataloader,
        loss_function,
        optimizer,
        scheduler,
        train_cfg,
        val_cfg,
        inferer,
        key_val_metric,
        val_post_pred_transforms=None,
        val_post_label_transforms=None,
        decollate=True,
    ):
        self._run_name = run_name
        self._device = device
        self._model = model.to(device)

        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader

        self._loss_function = loss_function
        self._optimizer = optimizer
        self._scheduler = scheduler

        self._inferer = inferer
        assert len(key_val_metric.values()) == 1
        for k, v in key_val_metric.items():
            self._key_val_metric_name = k
            self._key_val_metric_fn = v

        self._val_post_pred_transforms = val_post_pred_transforms
        self._val_post_label_transforms = val_post_label_transforms
        self._decollate = decollate

        self._train_cfg = self._init_loop_cfg(train_cfg)
        self._val_cfg = self._init_loop_cfg(val_cfg)

    def _init_loop_cfg(self, loop_cfg):
        if loop_cfg is None:
            return loop_cfg
        if "iter" not in loop_cfg:
            loop_cfg["iter"] = 0
        if ("max_epochs" in loop_cfg) and ("epoch" not in loop_cfg):
            loop_cfg["epoch"] = 0
        return loop_cfg

    def train(self, logger=None):
        if logger is None:
            logger = logging.getLogger()

        # init weights?
        best_metric = -1
        best_metric_epoch = -1
        # https://github.com/mlflow/mlflow/blob/master/examples/pytorch/logging/pytorch_log_model.ipynb
        with mlflow.start_run(run_name=self._run_name) as run:
            for epoch_idx in range(
                self._train_cfg["epoch"], self._train_cfg["max_epochs"]
            ):
                logger.info(
                    f"epoch {self._train_cfg['epoch'] + 1}/{self._train_cfg['max_epochs']}"
                )
                self._model.train()
                epoch_loss = 0
                step = 0
                for _, batch_data in enumerate(self._train_dataloader):
                    inputs, labels = batch_data["image"].to(self._device), batch_data[
                        "label"
                    ].to(self._device)
                    self._optimizer.zero_grad()
                    outputs = self._model(inputs)
                    loss = self._loss_function(outputs, labels)
                    loss.backward()
                    self._optimizer.step()
                    epoch_loss += loss.item()
                    mlflow.log_metric(
                        "train_iter_loss", loss.item(), step=self._train_cfg["iter"]
                    )
                    self._scheduler.step()
                    self._train_cfg["iter"] += 1
                    step += 1

                epoch_loss /= step
                logger.info(
                    f"epoch {self._train_cfg['epoch']} average loss: {epoch_loss:.4f}"
                )
                self._train_cfg["epoch"] += 1

                # validation
                if (epoch_idx + 1) % self._val_cfg["val_interval"] == 0:
                    metric = self.validate()
                    if metric > best_metric:
                        best_metric = metric
                        best_metric_epoch = epoch_idx
                        mlflow.pytorch.log_model(
                            self._model,
                            "model",
                        )
                        logger.info("saved new best metric model")
                    mlflow.log_metric(
                        key=self._key_val_metric_name, value=metric, step=epoch_idx
                    )
                    logger.info(
                        "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                            epoch_idx + 1, metric, best_metric, best_metric_epoch
                        )
                    )
        return run

    def validate(self):
        self._model.eval()

        for val_data in self._val_dataloader:
            inputs, labels = val_data["image"].to(self._device), val_data["label"].to(
                self._device
            )
            outputs = self._inferer(inputs, self._model)
            if self._decollate:
                outputs = decollate_batch(outputs)
                labels = decollate_batch(labels)
            outputs = [self._val_post_pred_transforms(i) for i in outputs]
            labels = [self._val_post_label_transforms(i) for i in labels]
            self._key_val_metric_fn(outputs, labels)

        metric = self._key_val_metric_fn.aggregate().item()
        self._key_val_metric_fn.reset()
        return metric
