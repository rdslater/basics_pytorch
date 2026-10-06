import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

# ---- Config ----
SEED = 0
BATCH_SIZE = 32
NUM_EPOCHS = 20
LR = 1e-3
HIDDEN = 64
CHECKPOINT = "best_digits_model.pt"


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
    data = load_digits()  # 1797 images of 8x8 pixels -> 64 features, 10 classes
    X_train, X_val, y_train, y_val = train_test_split(
        data.data, data.target, random_state=seed,
        stratify=data.target,  # keep class balance the same in both splits
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  # fit on train only
    X_val = scaler.transform(X_val)

    train_ds = TabularDataset(X_train, y_train, task="classification")
    val_ds = TabularDataset(X_val, y_val, task="classification")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    n_features = X_train.shape[1]
    n_classes = len(data.target_names)
    return train_loader, val_loader, n_features, n_classes


# ---- Model ----
def build_model(in_features, n_classes, hidden=HIDDEN):
    # Outputs raw scores (logits) for each class; no softmax here,
    # because CrossEntropyLoss applies log-softmax internally.
    return nn.Sequential(
        nn.Linear(in_features, hidden),
        nn.ReLU(),
        nn.Linear(hidden, n_classes),
    )


# ---- Train / eval ----
def run_epoch(model, loader, criterion, device, *, train, optimizer=None):
    """Run one pass over `loader`. Returns (avg loss, accuracy).

    `train` is keyword-only and required, so every call says which mode it is:
        run_epoch(..., train=True, optimizer=opt)   # training
        run_epoch(..., train=False)                 # validation
    """
    if train and optimizer is None:
        raise ValueError("train=True requires an optimizer")

    model.train(train)  # train(False) is the same as model.eval()
    total_loss, total_correct, n = 0.0, 0, 0

    with torch.set_grad_enabled(train):  # set_grad_enabled(False) == no_grad()
        for batch in loader:
            features = batch["features"].to(device)
            targets = batch["targets"].to(device)  # shape (B,), dtype long

            logits = model(features)  # shape (B, n_classes)
            loss = criterion(logits, targets)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_size = features.size(0)
            preds = logits.argmax(dim=1)  # predicted class per sample
            total_loss += loss.item() * batch_size
            total_correct += (preds == targets).sum().item()
            n += batch_size

    return total_loss / n, total_correct / n


def fit(model, train_loader, val_loader, criterion, optimizer, device,
        num_epochs=NUM_EPOCHS, checkpoint=CHECKPOINT):
    best_val_acc = 0.0
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device,
            train=True, optimizer=optimizer,
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, device,
            train=False,
        )

        print(f"Epoch {epoch:02d} | "
              f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint)
    return best_val_acc


# ---- Entry point ----
def main():
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader, n_features, n_classes = make_loaders()
    model = build_model(n_features, n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best = fit(model, train_loader, val_loader, criterion, optimizer, device)
    print(f"Best val accuracy: {best:.3f}")


if __name__ == "__main__":
    main()