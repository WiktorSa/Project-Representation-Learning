import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def count_correct(
    y_pred: torch.Tensor, y_true: torch.Tensor
):
    preds = torch.argmax(y_pred, dim=1)
    return (preds == y_true).float().sum()

def validate(
    model: nn.Module, 
    loss_fn: torch.nn.CrossEntropyLoss, 
    dataloader: DataLoader
):
    loss = 0
    correct = 0
    all = 0
    for X_batch, y_batch in dataloader:
        y_pred = model(X_batch.cuda())
        all += len(y_pred)
        loss += loss_fn(y_pred, y_batch.cuda()).sum()
        correct += count_correct(y_pred, y_batch.cuda())
    return loss / all, correct / all

def fit_classifier(
    model: nn.Module, optimiser: optim.Optimizer, 
    loss_fn: torch.nn.CrossEntropyLoss, train_dl: DataLoader, 
    val_dl: DataLoader, epochs: int, early_stop = None,
    print_metrics: str = True
):
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    for epoch in range(epochs):
        for X_batch, y_batch in tqdm(train_dl):
            y_pred = model(X_batch.cuda())
            loss = loss_fn(y_pred, y_batch.cuda())

            loss.backward()
            optimiser.step()
            optimiser.zero_grad()

        if print_metrics: 
            model.eval()
            with torch.no_grad():
                train_loss, train_acc = validate(
                    model=model, loss_fn=loss_fn, dataloader=train_dl
                ) 
                val_loss, val_acc = validate(
                    model=model, loss_fn=loss_fn, dataloader=val_dl
                )
                print(
                    f"Epoch {epoch}: "
                    f"train loss = {train_loss:.3f} (acc: {train_acc:.3f}), "
                    f"validation loss = {val_loss:.3f} (acc: {val_acc:.3f})"
                )
                
                train_losses.append(train_loss.cpu())
                train_accs.append(train_acc.cpu())
                val_losses.append(val_loss.cpu())
                val_accs.append(val_acc.cpu())

                if early_stop is not None:
                    if early_stop(val_loss, epoch, model, optimiser):
                        break
                
    if print_metrics:
        fig, axs = plt.subplots(2, 1, figsize=(24, 18))
        fig.suptitle('Fit info', fontsize=24)
        
        axs[0].plot(train_losses, label='Train')
        axs[0].plot(val_losses, label='Validation')
        axs[0].legend()
        axs[0].set_title("Loss", fontsize=18)
        axs[0].set_xlabel("Iteration", fontsize=14)
        axs[0].set_ylabel("Loss", fontsize=14)
        
        axs[1].plot(train_accs, label='Train')
        axs[1].plot(val_accs, label='Validation')
        axs[1].legend()
        axs[1].set_title("Accuracy", fontsize=18)
        axs[1].set_xlabel("Iteration", fontsize=14)
        axs[1].set_ylabel("Accuracy", fontsize=14)
        
        plt.show()
    
def calculate_metrics(model, val_dl, average='binary'):
    y_preds = []
    y_preds_argmax = []
    y_trues = []

    for X_batch, y_batch in val_dl:
        with torch.no_grad():
            y_pred = model(X_batch.cuda()).cpu()
            y_pred_argmax = torch.argmax(y_pred, dim=1)

            y_preds.extend(y_pred.tolist())
            y_preds_argmax.extend(y_pred_argmax.tolist())
            y_trues.extend(y_batch.tolist())

    accuracy = accuracy_score(y_trues, y_preds_argmax)
    prec = precision_score(y_trues, y_preds_argmax, average=average)
    rec = recall_score(y_trues, y_preds_argmax, average=average)
    f1 = f1_score(y_trues, y_preds_argmax, average=average)

    return {
        "Acccuracy": accuracy,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1
    }
