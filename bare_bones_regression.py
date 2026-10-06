import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
# ---- Setup ----
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)



class DataFrameDataset(Dataset):
    def __init__(self, feature_df, target_df, task="regression"):
        self.X = torch.tensor(feature_df.values, dtype=torch.float32)
        if task == "classification":
            target_dtype = torch.long
        else:
            target_dtype = torch.float32
        y = torch.tensor(target_df.to_numpy(), dtype=target_dtype)
        self.y = y.reshape(len(y), -1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {"features": self.X[idx], "targets": self.y[idx]}


model = nn.Sequential(
        nn.Linear(8, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    ).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def train_one_epoch(model, loader):
    model.train()
    total_loss, n = 0.0, 0

    for item in loader:
        features = item['features']
        targets = item['targets']
        features, targets = features.to(device), targets.to(device)

        optimizer.zero_grad()           # 1. clear old gradients
        logits = model(features)              # 2. forward pass
        loss = criterion(logits, targets)    # 3. compute loss
        loss.backward()                 # 4. backprop
        optimizer.step()                # 5. update weights

        total_loss += loss.item() * features.size(0)
        n += features.size(0)

    return total_loss / n


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss, n = 0.0, 0,

    for item in loader:
        features = item['features']
        targets = item['targets']
        features, targets = features.to(device), targets.to(device) 
        logits = model(features)
        loss = criterion(logits, targets)

        total_loss += loss.item() * features.size(0)
        n += features.size(0)

    return total_loss / n


def main():
    CA_data = fetch_california_housing(as_frame=True)
    features = CA_data['data']
    targets = CA_data['target']
    (
        train_features,
        val_features,
        train_targets,
        val_targets,
    ) = train_test_split(features, targets)
    scaling = StandardScaler()
    train_scaled = pd.DataFrame(scaling.fit_transform(train_features))
    val_scaled = pd.DataFrame(scaling.transform(val_features))
    train_ds = DataFrameDataset(train_scaled, train_targets, task="regression")
    val_ds = DataFrameDataset(val_scaled, val_targets)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    num_epochs = 20
    best_val_loss = float("inf")
    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader)
        val_loss = evaluate(model, val_loader)

        print(f"Epoch {epoch:02d} | "
              f"train loss {train_loss:.4f} | "
              f"val loss {val_loss:.4f} |")

        # Simple checkpointing  (Lightning: ModelCheckpoint callback)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")


if __name__ == "__main__":
    main()
