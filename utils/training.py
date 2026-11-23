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
WEIGHT_DECAY = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mixed precision on GPU: bf16 where the hardware has it (no loss scaling needed), else fp16.
# including_emulation=False: older GPUs (e.g. GTX 1080 Ti) only emulate bf16, which is slow.
_AMP_DTYPE = (torch.bfloat16 if device.type == "cuda"
              and torch.cuda.is_bf16_supported(including_emulation=False)
              else torch.float16)
_USE_AMP = device.type == "cuda"


def _autocast():
    return torch.autocast(device_type=device.type, dtype=_AMP_DTYPE, enabled=_USE_AMP)


def make_loader(X, y, shuffle, batch_size=BATCH_SIZE):
    ds = TensorDataset(torch.tensor(X), torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      pin_memory=device.type == "cuda")


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, preds = 0.0, []
    with torch.no_grad(), _autocast():
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            outputs = model(xb)
            total_loss += criterion(outputs.float(), yb).item() * len(yb)
            preds.append(outputs.argmax(dim=1).cpu())
    preds = torch.cat(preds).numpy()
    targets = loader.dataset.tensors[1].numpy()
    return total_loss / len(targets), (preds == targets).mean(), preds


def train_ft_transformer(X_train, y_train, X_val, y_val, num_classes, seed=42, verbose=True,
                         config=None, lr=LR, epochs=EPOCHS, patience=PATIENCE,
                         batch_size=BATCH_SIZE, weight_decay=WEIGHT_DECAY):
    """Trains with early stopping on validation loss and returns the best model."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = make_loader(X_train, y_train, shuffle=True, batch_size=batch_size)
    val_loader = make_loader(X_val, y_val, shuffle=False, batch_size=max(batch_size, 1024))

    model = FTTransformer(X_train.shape[1], num_classes, **(config or MODEL_CONFIG)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler(enabled=_USE_AMP and _AMP_DTYPE == torch.float16)

    best_val_loss, best_state, bad_epochs = float("inf"), None, 0

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

            optimizer.zero_grad()
            with _autocast():
                outputs = model(xb)
                loss = criterion(outputs.float(), yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

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
            if bad_epochs >= patience:
                if verbose:
                    print(f"[INFO] Early stopping after epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    model.eval()
    model.best_val_loss = best_val_loss
    return model


def predict_proba(model, X, batch_size=4096):
    model_device = next(model.parameters()).device
    out = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.tensor(X[start:start + batch_size]).to(model_device)
            out.append(torch.softmax(model(xb).float(), dim=1).cpu())
    return torch.cat(out).numpy()
