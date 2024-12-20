import os
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from src.dataset import RAVDESSDataset
from src.model import LSTM_RNN_Model
from src.train import train
from src.generate_labels import extract_labels
from config import PROCESSED_DATA_DIR, LABELS_FILE, RAW_DATA_DIR

def main():
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # Step 1: Generate labels.csv if not already present
    if not os.path.exists(LABELS_FILE):
        print("Generating labels.csv...")
        extract_labels(RAW_DATA_DIR, LABELS_FILE)
    else:
        print("labels.csv already exists.")

    ravdess_dataset = RAVDESSDataset(labels_csv=LABELS_FILE, raw_data_dir=RAW_DATA_DIR)
    labels = ravdess_dataset.get_all_labels(as_one_hot=False)
    
    # Create a list of dataset indices
    indices = list(range(len(ravdess_dataset)))
    # Perform the first split: train and temp (cross-val + test)
    train_idx, temp_idx = train_test_split(indices, test_size=0.4, stratify=labels, random_state=1234)
    
    # Extract labels for the temporary dataset
    temp_labels = [labels[i] for i in temp_idx]
    # Perform the second split: cross-validation and test
    cross_val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=temp_labels, random_state=1234)

    # Create dataset subsets using Subset
    train_dataset = Subset(ravdess_dataset, train_idx)
    cross_val_dataset = Subset(ravdess_dataset, cross_val_idx)
    test_dataset = Subset(ravdess_dataset, test_idx)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    cross_val_loader = DataLoader(cross_val_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Model setup
    model = LSTM_RNN_Model(input_size=40, lstm_hidden_size=64, rnn_hidden_size=32, num_classes=8)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # Train
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Training with {device} device.')
    train(model, train_loader, cross_val_loader, criterion, optimizer, num_epochs=100, device=device)

if __name__ == "__main__":
    main()
