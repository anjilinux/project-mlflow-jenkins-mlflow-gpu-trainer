import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim

device = "cuda" if torch.cuda.is_available() else "cpu"

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

model = SimpleNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

mlflow.set_experiment("GPU_MLflow_Jenkins")

with mlflow.start_run():
    for epoch in range(5):
        x = torch.randn(32, 10).to(device)
        y = torch.randn(32, 1).to(device)

        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()

        mlflow.log_metric("loss", loss.item(), step=epoch)
        print(f"Epoch {epoch} | Loss {loss.item()}")

    mlflow.pytorch.log_model(model, "model")
    mlflow.log_param("device", device)
