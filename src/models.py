"""
Model definitions for seismic data processing using PyTorch.
"""

import torch
import torch.nn as nn


class SeismicCNN(nn.Module):
    """
    A simple CNN model for seismic waveform classification.
    
    This is a template model that can be extended for specific tasks
    such as landslide detection, earthquake classification, etc.
    
    Args:
        input_channels (int): Number of input channels (default: 3 for 3-component seismogram)
        num_classes (int): Number of output classes
        dropout (float): Dropout rate for regularization
    """
    
    def __init__(self, input_channels=3, num_classes=2, dropout=0.5):
        super(SeismicCNN, self).__init__()
        
        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv1d(input_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            
            # Second convolutional block
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            
            # Third convolutional block
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, length)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        x = self.features(x)
        x = self.classifier(x)
        return x


def load_model(checkpoint_path, model_class=SeismicCNN, device='cpu', **model_kwargs):
    """
    Load a trained model from a checkpoint.
    
    Args:
        checkpoint_path (str): Path to the model checkpoint (.pt or .pth file)
        model_class: The model class to instantiate
        device (str): Device to load the model on ('cpu' or 'cuda')
        **model_kwargs: Additional arguments to pass to the model constructor
        
    Returns:
        model: The loaded model
    """
    model = model_class(**model_kwargs)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model


def save_model(model, filepath, optimizer=None, epoch=None, loss=None, metadata=None):
    """
    Save a model checkpoint.
    
    Args:
        model (nn.Module): The model to save
        filepath (str): Path where to save the checkpoint
        optimizer: Optional optimizer state to save
        epoch (int): Optional current epoch number
        loss (float): Optional current loss value
        metadata (dict): Optional additional metadata to save
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if epoch is not None:
        checkpoint['epoch'] = epoch
    if loss is not None:
        checkpoint['loss'] = loss
    if metadata is not None:
        checkpoint['metadata'] = metadata
    
    torch.save(checkpoint, filepath)
