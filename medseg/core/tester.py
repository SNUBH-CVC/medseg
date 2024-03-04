from ignite.engine import Events, create_supervised_evaluator


class SupervisedTestWrapper:

    def __init__(self, model, device, dataloader, evaluator_kwargs):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.evaluator = create_supervised_evaluator(
            model,
            device=device,
            **evaluator_kwargs,
        )
        self.evaluator.add_event_handler(
            Events.ITERATION_COMPLETED, self.on_test_iteration_completed
        )
        self.evaluator.add_event_handler(Events.EPOCH_COMPLETED, self.on_test_completed)

    def on_test_iteration_completed(self, engine):
        print("test iteration completed")

    def on_test_completed(self, engine):
        print("test completed")

    def run(self):
        self.evaluator.run(self.dataloader)
