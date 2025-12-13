"""
Example training script for seismic data classification.

This script demonstrates a complete training pipeline including:
- Data loading and preprocessing
- Model training with validation
- Checkpointing
- Visualization of results
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.models import SeismicCNN, save_model
from src.data import SeismicDataset, SeismicDataProcessor
from src.utils import set_seed, get_device, ensure_dir, plot_training_history


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for data, labels in dataloader:
        data, labels = data.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * data.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            
            outputs = model(data)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * data.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def main(args):
    """Main training function."""
    # Set seed
    set_seed(args.seed)
    
    # Get device
    device = get_device()
    
    # Create output directories
    ensure_dir('checkpoints')
    ensure_dir('plots')
    
    # Initialize data processor
    processor = SeismicDataProcessor(
        sampling_rate=args.sampling_rate,
        window_length=args.window_length,
        normalize=True
    )
    
    # TODO: Replace with your actual data
    # For this example, we assume you have a list of data files and labels
    # data_files = ['path/to/file1.mseed', 'path/to/file2.mseed', ...]
    # labels = [0, 1, 0, ...]
    # dataset = SeismicDataset(data_files, labels, processor)
    
    print("Note: This is a template script. Please update with your actual data loading logic.")
    print("See notebooks/example_analysis.ipynb for examples with synthetic data.")
    
    # Split dataset into train and validation
    # train_size = int(0.8 * len(dataset))
    # val_size = len(dataset) - train_size
    # train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize model
    model = SeismicCNN(
        input_channels=args.input_channels,
        num_classes=args.num_classes,
        dropout=args.dropout
    ).to(device)
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        # Note: Uncomment when you have actual data
        # train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        # val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # history['train_loss'].append(train_loss)
        # history['train_acc'].append(train_acc)
        # history['val_loss'].append(val_loss)
        # history['val_acc'].append(val_acc)
        
        # print(f"Epoch {epoch+1}/{args.epochs}")
        # print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        # print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # scheduler.step(val_loss)
        
        # Save best model
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     save_model(
        #         model,
        #         f'checkpoints/best_model_epoch{epoch+1}.pth',
        #         optimizer=optimizer,
        #         epoch=epoch,
        #         loss=val_loss,
        #         metadata={'val_acc': val_acc}
        #     )
        #     print(f"  Saved best model (val_loss: {val_loss:.4f})")
        
        pass  # Remove this when uncommenting the training loop
    
    # Save final model
    # save_model(model, 'checkpoints/final_model.pth', optimizer=optimizer, epoch=args.epochs)
    
    # Plot training history
    # plot_training_history(history, save_path='plots/training_history.png')
    
    print("\nTraining template complete!")
    print("Update this script with your data loading logic to begin actual training.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train seismic classification model')
    
    # Data parameters
    parser.add_argument('--sampling_rate', type=float, default=100.0, help='Sampling rate (Hz)')
    parser.add_argument('--window_length', type=float, default=30.0, help='Window length (seconds)')
    parser.add_argument('--input_channels', type=int, default=3, help='Number of input channels')
    
    # Model parameters
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    main(args)
