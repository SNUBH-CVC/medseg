from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import mlflow
from ignite.engine import Engine, Events
from monai.handlers.mlflow_handler import DEFAULT_TAG, MLFlowHandler
from monai.utils import CommonKeys
from torch.utils.data import Dataset


class MedSegMLFlowHandler(MLFlowHandler):

    def __init__(
        self,
        tracking_uri: str | None = None,
        iteration_log: bool | Callable[[Engine, int], bool] = True,
        epoch_log: bool | Callable[[Engine, int], bool] = True,
        epoch_logger: Callable[[Engine], Any] | None = None,
        iteration_logger: Callable[[Engine], Any] | None = None,
        dataset_logger: Callable[[Mapping[str, Dataset]], Any] | None = None,
        dataset_dict: Mapping[str, Dataset] | None = None,
        dataset_keys: str = CommonKeys.IMAGE,
        output_transform: Callable = lambda x: x[0],
        global_epoch_transform: Callable = lambda x: x,
        state_attributes: Sequence[str] | None = None,
        tag_name: str = DEFAULT_TAG,
        experiment_name: str = "monai_experiment",
        run_name: str | None = None,
        experiment_param: dict | None = None,
        artifacts: str | Sequence[Path] | None = None,
        optimizer_param_names: str | Sequence[str] = "lr",
        close_on_complete: bool = False,
        save_dict: dict | None = None,
        key_metric_name: str | None = None,
        log_model: bool = False,
        artifacts_at_start: dict | None = None,
    ) -> None:
        super().__init__(
            tracking_uri,
            iteration_log,
            epoch_log,
            epoch_logger,
            iteration_logger,
            dataset_logger,
            dataset_dict,
            dataset_keys,
            output_transform,
            global_epoch_transform,
            state_attributes,
            tag_name,
            experiment_name,
            run_name,
            experiment_param,
            artifacts,
            optimizer_param_names,
            close_on_complete,
        )
        if log_model:
            assert save_dict is not None and key_metric_name is not None
        self.save_dict = save_dict
        self.key_metric_name = key_metric_name
        self.log_model = log_model
        self.best_metric = -1
        self.artifacts_at_start = artifacts_at_start

    def attach(self, engine: Engine) -> None:
        super().attach(engine)
        if self.log_model:
            engine.add_event_handler(Events.EPOCH_COMPLETED, self._save_checkpoint)
        if self.artifacts_at_start is not None:
            engine.add_event_handler(Events.STARTED, self._save_artifacts_at_start)

    def _save_checkpoint(self, engine: Engine) -> None:
        metric = engine.state.metrics[self.key_metric_name]
        if metric > self.best_metric:
            self.best_metric = metric
            # MlflowClient doesn't support log_model
            with mlflow.start_run(run_id=self.cur_run.info.run_id):
                mlflow.pytorch.log_model(self.save_dict["model"], "model")

    def _save_artifacts_at_start(self, engine: Engine) -> None:
        if self.artifacts_at_start and self.cur_run:
            for local_path, artifact_path in self.artifacts_at_start.items():
                self.client.log_artifact(
                    self.cur_run.info.run_id, local_path, artifact_path
                )
