
def test_pytorch():
    import torch
    x = torch.randn(32, 784)
    y = torch.randn(32, 10)

    model = torch.nn.Sequential(
        torch.nn.Linear(784, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 10),
    )

    loss_fn = torch.nn.MSELoss(reduction='sum')

    learning_rate = 1e-4
    for t in range(3):
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        print(t, loss.item())

        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            for param in model.parameters():
                param.data -= learning_rate * param.grad
