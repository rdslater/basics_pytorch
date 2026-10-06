import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

# ---- Setup ----
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

# Dummy data: 1000 samples, 20 features, 3 classes
X = torch.randn(1000, 20)
y = torch.randint(0, 3, (1000,))
train_ds, val_ds = random_split(TensorDataset(X, y), [800, 200])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False)

# ---- Model, loss, optimizer ----  (Lightning: __init__ + configure_optimizers)
model = nn.Sequential(
    nn.Linear(20, 64),
    nn.ReLU(),
    nn.Linear(64, 3),
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def train_one_epoch(model, loader):
    model.train()                      
    total_loss, correct, n = 0.0, 0, 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()           # 1. clear old gradients
        logits = model(xb)              # 2. forward pass
        loss = criterion(logits, yb)    # 3. compute loss
        loss.backward()                 # 4. backprop
        optimizer.step()                # 5. update weights

        total_loss += loss.item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        n += xb.size(0)

    return total_loss / n, correct / n


@torch.no_grad()                       
def evaluate(model, loader):
    model.eval()                        
    total_loss, correct, n = 0.0, 0, 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)

        total_loss += loss.item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        n += xb.size(0)

    return total_loss / n, correct / n

def main():
    num_epochs = 10
    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader)
        val_loss, val_acc = evaluate(model, val_loader)

        print(f"Epoch {epoch:02d} | "
            f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.3f}")

        # Simple checkpointing  (Lightning: ModelCheckpoint callback)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")

if __name__=="__main__":
    main()