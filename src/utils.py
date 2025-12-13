"""
Utility functions for visualization, training, and evaluation.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path


def plot_seismogram(data, sampling_rate=100.0, title="Seismogram", labels=None, save_path=None):
    """
    Plot seismic waveform data.
    
    Args:
        data (np.ndarray or torch.Tensor): Seismic data of shape (n_channels, n_samples)
        sampling_rate (float): Sampling rate in Hz
        title (str): Plot title
        labels (list): Optional labels for each channel
        save_path (str): Optional path to save the figure
    """
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    
    n_channels, n_samples = data.shape
    time = np.arange(n_samples) / sampling_rate
    
    fig, axes = plt.subplots(n_channels, 1, figsize=(12, 2 * n_channels), sharex=True)
    
    if n_channels == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        ax.plot(time, data[i], 'k', linewidth=0.5)
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        
        if labels and i < len(labels):
            ax.set_ylabel(f'{labels[i]}\nAmplitude')
    
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def plot_spectrogram(data, sampling_rate=100.0, title="Spectrogram", save_path=None):
    """
    Plot spectrogram of seismic data.
    
    Args:
        data (np.ndarray or torch.Tensor): Seismic data of shape (n_channels, n_samples)
        sampling_rate (float): Sampling rate in Hz
        title (str): Plot title
        save_path (str): Optional path to save the figure
    """
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    
    n_channels = data.shape[0]
    
    fig, axes = plt.subplots(n_channels, 1, figsize=(12, 3 * n_channels))
    
    if n_channels == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        ax.specgram(data[i], Fs=sampling_rate, cmap='viridis', NFFT=256, noverlap=128)
        ax.set_ylabel('Frequency (Hz)')
        ax.set_xlabel('Time (s)')
        ax.set_title(f'Channel {i+1}')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def plot_training_history(history, save_path=None):
    """
    Plot training history (loss and metrics).
    
    Args:
        history (dict): Dictionary containing 'train_loss', 'val_loss', etc.
        save_path (str): Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    if 'train_loss' in history:
        axes[0].plot(history['train_loss'], label='Train Loss')
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracy or other metrics
    if 'train_acc' in history:
        axes[1].plot(history['train_acc'], label='Train Acc')
    if 'val_acc' in history:
        axes[1].plot(history['val_acc'], label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def ensure_dir(directory):
    """
    Ensure a directory exists, create if it doesn't.
    
    Args:
        directory (str or Path): Directory path
        
    Returns:
        Path: Path object for the directory
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed=42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model (nn.Module): PyTorch model
        
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device():
    """
    Get the best available device (CUDA if available, otherwise CPU).
    
    Returns:
        torch.device: The device to use
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device
