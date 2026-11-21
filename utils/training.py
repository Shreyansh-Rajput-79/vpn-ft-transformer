import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.ft_transformer import FTTransformer

MODEL_CONFIG = dict(d_model=64, n_heads=4, n_layers=3, dim_feedforward=128, dropout=0.1)
EPOCHS = 60
PATIENCE = 10
BATCH_SIZE = 128
LR = 3e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loader(X, y, shuffle):
    ds = TensorDataset(torch.tensor(X), torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, preds = 0.0, []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            total_loss += criterion(outputs, yb).item() * len(yb)
            preds.append(outputs.argmax(dim=1).cpu())
    preds = torch.cat(preds).numpy()
    targets = loader.dataset.tensors[1].numpy()
    return total_loss / len(targets), (preds == targets).mean(), preds


def train_ft_transformer(X_train, y_train, X_val, y_val, num_classes, seed=42, verbose=True):
    """Trains with early stopping on validation loss and returns the best model."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader = make_loader(X_val, y_val, shuffle=False)

    model = FTTransformer(X_train.shape[1], num_classes, **MODEL_CONFIG).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    best_val_loss, best_state, bad_epochs = float("inf"), None, 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(yb)
            correct += (outputs.argmax(dim=1) == yb).sum().item()
            total += len(yb)

        val_loss, val_acc, _ = evaluate(model, val_loader, criterion)
        if verbose:
            print(f"[EPOCH {epoch+1:2d}] train loss {total_loss/total:.4f} acc {correct/total:.4f} | "
                  f"val loss {val_loss:.4f} acc {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss, best_state, bad_epochs = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                if verbose:
                    print(f"[INFO] Early stopping after epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    model.eval()
    return model


def predict_proba(model, X):
    with torch.no_grad():
        return torch.softmax(model(torch.tensor(X).to(device)), dim=1).cpu().numpy()
