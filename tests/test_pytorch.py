def test_run():
    import torch
    import torch.utils.data
    import torch.nn.functional as F
    import ignite
    import tqdm

    x = torch.randn(32, 784)
    y = torch.randint(10, (32,))
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y), batch_size=16, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y), batch_size=16, shuffle=False
    )

    model = torch.nn.Sequential(
        torch.nn.Linear(784, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 10),
        torch.nn.Softmax(),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    trainer = ignite.engine.create_supervised_trainer(
        model, optimizer, F.nll_loss, device=device
    )
    evaluator = ignite.engine.create_supervised_evaluator(
        model,
        metrics={
            "accuracy": ignite.metrics.Accuracy(),
            "nll": ignite.metrics.Loss(F.nll_loss),
        },
        device=device,
    )

    desc = "ITERATION - loss: {:.2f}"
    pbar = tqdm.tqdm(
        initial=0, leave=False, total=len(train_loader), desc=desc.format(0)
    )
    log_interval = 1

    @trainer.on(ignite.engine.Events.ITERATION_COMPLETED)
    def _log_training_loss(engine):
        it = (engine.state.iteration - 1) % len(train_loader) + 1
        if it % log_interval == 0:
            pbar.desc = desc.format(engine.state.output)
            pbar.update(log_interval)

    @trainer.on(ignite.engine.Events.EPOCH_COMPLETED)
    def _log_training_results(engine):
        pbar.refresh()
        evaluator.run(train_loader)
        avg_accuracy = evaluator.state.metrics["accuracy"]
        avg_nll = evaluator.state.metrics["nll"]
        tqdm.tqdm.write(
            "Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}".format(
                engine.state.epoch, avg_accuracy, avg_nll
            )
        )

    @trainer.on(ignite.engine.Events.EPOCH_COMPLETED)
    def _log_validation_results(engine):
        evaluator.run(val_loader)
        avg_accuracy = evaluator.state.metrics["accuracy"]
        avg_nll = evaluator.state.metrics["nll"]
        tqdm.tqdm.write(
            "Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}".format(
                engine.state.epoch, avg_accuracy, avg_nll
            )
        )

        pbar.n = pbar.last_print_n = 0

    trainer.run(train_loader, max_epochs=2)
    pbar.close()
