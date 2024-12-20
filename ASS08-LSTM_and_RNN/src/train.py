import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from config import CHECKPOINTS_DIR
from src.utils import save_model_checkpoint

def train(model, train_loader, cross_val_loader, criterion, optimizer, num_epochs, device):
    model.to(device)
    best_val_loss = float("inf")
    best_model_path = os.path.join(CHECKPOINTS_DIR, "best_model.pth")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

        # Cross-validation phase
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for inputs, labels in cross_val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                
                # Compute loss
                loss = criterion(outputs, labels.argmax(dim=1))
                val_loss += loss.item()
        
        val_loss /= len(cross_val_loader)
        print(f"Validation Loss: {val_loss:.4f}")
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"New best model found! Validation Loss: {val_loss:.4f}")
            save_model_checkpoint(model, best_model_path)
        else:
            print("No improvement.")

    print(f"Training complete. Best Validation Loss: {best_val_loss:.4f}")