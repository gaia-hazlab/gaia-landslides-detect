"""
Unit tests for model definitions.

To run these tests:
    pytest tests/test_models.py
"""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models import SeismicCNN, save_model, load_model


class TestSeismicCNN:
    """Tests for SeismicCNN model."""
    
    def test_model_creation(self):
        """Test that model can be created with default parameters."""
        model = SeismicCNN()
        assert model is not None
        assert isinstance(model, torch.nn.Module)
    
    def test_model_forward(self):
        """Test forward pass with dummy data."""
        model = SeismicCNN(input_channels=3, num_classes=2)
        batch_size = 4
        n_channels = 3
        n_samples = 3000
        
        # Create dummy input
        x = torch.randn(batch_size, n_channels, n_samples)
        
        # Forward pass
        output = model(x)
        
        # Check output shape
        assert output.shape == (batch_size, 2)
    
    def test_model_with_different_channels(self):
        """Test model with different number of input channels."""
        for n_channels in [1, 3, 6]:
            model = SeismicCNN(input_channels=n_channels, num_classes=2)
            x = torch.randn(2, n_channels, 3000)
            output = model(x)
            assert output.shape == (2, 2)
    
    def test_model_with_different_classes(self):
        """Test model with different number of output classes."""
        for n_classes in [2, 5, 10]:
            model = SeismicCNN(input_channels=3, num_classes=n_classes)
            x = torch.randn(2, 3, 3000)
            output = model(x)
            assert output.shape == (2, n_classes)
    
    def test_model_eval_mode(self):
        """Test that model can be set to evaluation mode."""
        model = SeismicCNN()
        model.eval()
        assert not model.training
    
    def test_model_train_mode(self):
        """Test that model can be set to training mode."""
        model = SeismicCNN()
        model.train()
        assert model.training


class TestModelSaveLoad:
    """Tests for model save/load functionality."""
    
    def test_save_and_load_model(self, tmp_path):
        """Test saving and loading a model."""
        # Create and save model
        model = SeismicCNN(input_channels=3, num_classes=2)
        save_path = tmp_path / "test_model.pth"
        save_model(model, str(save_path))
        
        # Check file exists
        assert save_path.exists()
        
        # Load model
        loaded_model = load_model(
            str(save_path),
            model_class=SeismicCNN,
            device='cpu',
            input_channels=3,
            num_classes=2
        )
        
        assert loaded_model is not None
        assert isinstance(loaded_model, SeismicCNN)
    
    def test_save_load_consistency(self, tmp_path):
        """Test that loaded model produces same output as original."""
        # Create model and dummy input
        model = SeismicCNN(input_channels=3, num_classes=2)
        x = torch.randn(1, 3, 3000)
        
        # Get output from original model
        model.eval()
        with torch.no_grad():
            original_output = model(x)
        
        # Save and load model
        save_path = tmp_path / "test_model.pth"
        save_model(model, str(save_path))
        loaded_model = load_model(
            str(save_path),
            model_class=SeismicCNN,
            device='cpu',
            input_channels=3,
            num_classes=2
        )
        
        # Get output from loaded model
        with torch.no_grad():
            loaded_output = loaded_model(x)
        
        # Check outputs are the same
        assert torch.allclose(original_output, loaded_output, atol=1e-6)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
