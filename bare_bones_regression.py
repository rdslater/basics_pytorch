import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
# ---- Setup ----
device = "cuda" if torch.cuda.is_available() else "cpu"
# Seed still seems to be inconsistent Need to explore more
torch.manual_seed(0)
"""
This is the standard apprach that is very explicit in what happens 
in the training loop. Pretty standard stuff
"""


# 1. Create Your Dataset
class DataFrameDataset(Dataset):
    """Takes is a features dataframe and target dataframe
        Converts both to tensors and fixes the shape as targets
        being series tend to get "mishapen" by the lack of a second dimension
        I like to return a dictionary as many times in my work we want to know
        which example is being studied, and that info is invariably in the file name
        Also allows you to pass more "stuff" out for analysis
    """
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


"""
Nothing wild here--basic neural network.  Notice NO ACTIVATION
After the final layer as the loss functions typically have that included
A little different from Tensorflow
"""

model = nn.Sequential(
        nn.Linear(8, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    ).to(device)

"""
Choose some basic examples
Mean Sqaured Error Loss
Adam optimizer with lr of 0.001
"""

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def train_one_epoch(model, loader):
    """
    This is both a blessing an a curse.
    Rather than a fit method you control the behavior manually
    Very much like old school tensorflow 1.x
    You can see the basic steps involved
    """
    model.train()  # turn on dropout and learning
    total_loss, n = 0.0, 0

    for item in loader:
        features = item['features']
        targets = item['targets']
        # In torch you need to put data on the right device!
        features, targets = features.to(device), targets.to(device)
        # Your basic look. You can reuse this for 95% of you data
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
    """
    Basic Repeat of the train loop but turns off learning (the decorator)
    and dropout
    Model is in pure evaluation mode
    """
    
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
    """
    Put together all the pieces about
    """
    # 1. Setup the data
    CA_data = fetch_california_housing(as_frame=True)
    features = CA_data['data']
    targets = CA_data['target']
    # 2. Split the data
    (
        train_features,
        val_features,
        train_targets,
        val_targets,
    ) = train_test_split(features, targets)
    # 3. Scale the data
    scaling = StandardScaler()
    train_scaled = pd.DataFrame(scaling.fit_transform(train_features))
    val_scaled = pd.DataFrame(scaling.transform(val_features))
    # 4. Setup the dataloaders (and datasets)
    train_ds = DataFrameDataset(train_scaled, train_targets, task="regression")
    val_ds = DataFrameDataset(val_scaled, val_targets)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    num_epochs = 20
    best_val_loss = float("inf")
    # Swtich to Human Counting (start at 1 and not zero)
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
