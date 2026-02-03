import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
from mlflow.models.signature import infer_signature

# -------------------------
# MLflow Setup
# -------------------------
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5555"))
mlflow.set_experiment("GPU_MLflow_Jenkins")

# -------------------------
# Reproducibility
# -------------------------
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# -------------------------
# Device
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Model
# -------------------------
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

model = SimpleNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

epochs = 5
batch_size = 32

# -------------------------
# Training
# -------------------------
with mlflow.start_run():

    mlflow.log_params({
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": 0.001,
        "optimizer": "Adam",
        "device": device,
        "gpu_available": torch.cuda.is_available()
    })

    if torch.cuda.is_available():
        mlflow.log_param("gpu_name", torch.cuda.get_device_name(0))

    start = time.time()

    for epoch in range(epochs):
        x = torch.randn(batch_size, 10).to(device)
        y = torch.randn(batch_size, 1).to(device)

        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()

        mlflow.log_metric("loss", loss.item(), step=epoch)
        print(f"Epoch {epoch} | Loss {loss.item()}")

    # -------------------------
    # Model Logging
    # -------------------------
    example_input = torch.randn(1, 10).to(device)

    signature = infer_signature(
        example_input.cpu().numpy(),
        model(example_input).detach().cpu().numpy()
    )

    mlflow.pytorch.log_model(
        model,
        "model",
        signature=signature,
        input_example=example_input.cpu().numpy()
    )

    mlflow.log_metric("training_time", time.time() - start)
