import os
from os.path import join
import torch.nn as nn
import torch.optim

class EarlyStopping:
    def __init__(self, model_dir: str="model", patience: int = 7):
        self.counter = 0
        self.patience = patience
        self.model_dir = model_dir
        
        os.makedirs(model_dir, exist_ok=True)
        self.best_val_loss = float("inf")
        self.file_name = "model.pt"
        
    def __call__(self, val_loss: float, epoch: int, model: nn.Module, optimizer: torch.optim) -> bool:
        # Save best weights
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.counter = 0
            
            torch.save(
                obj={
                  'epoch': epoch,
                  'loss': val_loss,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict()
                },
                f=join(self.model_dir, self.file_name)
            )
            
            return False
        
        # Load best weights when validation loss won't improve
        elif self.counter >= self.patience:
            print(f"Val loss didn't improve in {self.patience} iterations. Early stopping.")
            checkpoint = torch.load(join(self.model_dir, self.file_name))
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            return True

        # Increase counter when validation is not improving
        else:
            self.counter += 1
            return False
