import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

# ---- Config ----
SEED = 0
BATCH_SIZE = 32
NUM_EPOCHS = 20
LR = 1e-3
HIDDEN = 64
CHECKPOINT = "best_model.pt"


# ---- Data ----
class TabularDataset(Dataset):
    """Wraps features/targets (DataFrame, Series or ndarray) as tensors."""

    def __init__(self, features, targets, task="regression"):
        self.X = torch.as_tensor(np.asarray(features), dtype=torch.float32)
        y = np.asarray(targets)
        if task == "classification":
            self.y = torch.as_tensor(y, dtype=torch.long).view(-1)
        else:
            self.y = torch.as_tensor(y, dtype=torch.float32).reshape(len(y), -1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {"features": self.X[idx], "targets": self.y[idx]}


def make_loaders(batch_size=BATCH_SIZE, seed=SEED):
    data = fetch_california_housing(as_frame=True)
    X_train, X_val, y_train, y_val = train_test_split(
        data.data, data.target, random_state=seed
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  # fit on train only
    X_val = scaler.transform(X_val)

    train_loader = DataLoader(
        TabularDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TabularDataset(X_val, y_val), batch_size=batch_size, shuffle=False
    )
    return train_loader, val_loader, X_train.shape[1]


# ---- Model ----
def build_model(in_features, hidden=HIDDEN, out_features=1):
    return nn.Sequential(
        nn.Linear(in_features, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_features),
    )


# ---- Train / eval ----
def run_epoch(model, loader, criterion, device, optimizer=None):
    """Train if an optimizer is given, otherwise evaluate. Returns avg loss."""
    training = optimizer is not None
    model.train(training)
    total_loss, n = 0.0, 0

    with torch.set_grad_enabled(training):
        for batch in loader:
            features = batch["features"].to(device)
            targets = batch["targets"].to(device)

            preds = model(features)
            loss = criterion(preds, targets)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * features.size(0)
            n += features.size(0)

    return total_loss / n


def fit(model, train_loader, val_loader, criterion, optimizer, device,
        num_epochs=NUM_EPOCHS, checkpoint=CHECKPOINT):
    best_val_loss = float("inf")
    for epoch in range(1, num_epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = run_epoch(model, val_loader, criterion, device)

        print(f"Epoch {epoch:02d} | train {train_loss:.4f} | val {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint)
    return best_val_loss


# ---- Entry point ----
def main():
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader, n_features = make_loaders()
    model = build_model(n_features).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best = fit(model, train_loader, val_loader, criterion, optimizer, device)
    print(f"Best val loss: {best:.4f}")


if __name__ == "__main__":
    main()