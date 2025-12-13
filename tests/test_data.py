"""
Unit tests for data processing utilities.

To run these tests:
    pytest tests/test_data.py
"""

import pytest
import numpy as np
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data import SeismicDataProcessor, SeismicDataset


class TestSeismicDataProcessor:
    """Tests for SeismicDataProcessor class."""
    
    def test_processor_creation(self):
        """Test processor can be created with default parameters."""
        processor = SeismicDataProcessor()
        assert processor is not None
        assert processor.sampling_rate == 100.0
        assert processor.window_length == 30.0
        assert processor.normalize is True
    
    def test_processor_custom_params(self):
        """Test processor with custom parameters."""
        processor = SeismicDataProcessor(
            sampling_rate=50.0,
            window_length=60.0,
            normalize=False
        )
        assert processor.sampling_rate == 50.0
        assert processor.window_length == 60.0
        assert processor.normalize is False
        assert processor.n_samples == 3000
    
    def test_normalize_std(self):
        """Test standardization normalization."""
        processor = SeismicDataProcessor()
        data = np.random.randn(3, 1000) * 5 + 10
        
        normalized = processor._normalize(data, method='std')
        
        # Check mean is close to 0 and std is close to 1
        assert np.allclose(np.mean(normalized, axis=1), 0, atol=1e-6)
        assert np.allclose(np.std(normalized, axis=1), 1, atol=1e-6)
    
    def test_normalize_minmax(self):
        """Test min-max normalization."""
        processor = SeismicDataProcessor()
        data = np.random.randn(3, 1000) * 5 + 10
        
        normalized = processor._normalize(data, method='minmax')
        
        # Check values are in [0, 1] range
        assert np.all(normalized >= 0)
        assert np.all(normalized <= 1)
    
    def test_normalize_maxabs(self):
        """Test max absolute value normalization."""
        processor = SeismicDataProcessor()
        data = np.random.randn(3, 1000) * 5
        
        normalized = processor._normalize(data, method='maxabs')
        
        # Check max absolute value is 1
        for i in range(3):
            assert np.abs(normalized[i]).max() <= 1.0
    
    def test_to_torch(self):
        """Test numpy to torch conversion."""
        processor = SeismicDataProcessor()
        data = np.random.randn(3, 1000)
        
        tensor = processor.to_torch(data)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.float32
        assert tensor.shape == (3, 1000)
        assert torch.allclose(tensor, torch.from_numpy(data).float())
    
    def test_stream_to_array_padding(self):
        """Test that short data is padded correctly."""
        processor = SeismicDataProcessor(sampling_rate=100.0, window_length=10.0)
        
        # Create mock stream with short data
        from unittest.mock import Mock
        mock_trace = Mock()
        mock_trace.data = np.random.randn(500)  # Shorter than n_samples
        mock_stream = [mock_trace] * 3
        
        result = processor.stream_to_array(mock_stream, n_components=3)
        
        assert result.shape == (3, 1000)  # Should be padded to full length


class TestSeismicDataset:
    """Tests for SeismicDataset class."""
    
    def test_dataset_creation(self):
        """Test dataset can be created."""
        data_list = [np.random.randn(3, 1000) for _ in range(10)]
        labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        
        dataset = SeismicDataset(data_list, labels)
        
        assert len(dataset) == 10
    
    def test_dataset_getitem(self):
        """Test getting items from dataset."""
        data_list = [np.random.randn(3, 1000) for _ in range(5)]
        labels = [0, 1, 0, 1, 0]
        
        dataset = SeismicDataset(data_list, labels)
        
        data, label = dataset[0]
        
        assert isinstance(data, torch.Tensor)
        assert data.shape == (3, 1000)
        assert label == 0
    
    def test_dataset_without_labels(self):
        """Test dataset without labels."""
        data_list = [np.random.randn(3, 1000) for _ in range(5)]
        
        dataset = SeismicDataset(data_list, labels=None)
        
        data = dataset[0]
        
        assert isinstance(data, torch.Tensor)
        assert data.shape == (3, 1000)
    
    def test_dataset_with_transform(self):
        """Test dataset with custom transform."""
        data_list = [np.random.randn(3, 1000) for _ in range(5)]
        
        # Simple transform that multiplies by 2
        def transform(x):
            return x * 2
        
        dataset = SeismicDataset(data_list, labels=None, transform=transform)
        
        original_data = data_list[0]
        transformed_data = dataset[0]
        
        expected = torch.from_numpy(original_data).float() * 2
        assert torch.allclose(transformed_data, expected)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
