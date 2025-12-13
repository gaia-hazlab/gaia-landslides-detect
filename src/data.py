"""
Data processing utilities for seismic data using ObsPy and SeisBench.
"""

import numpy as np
import torch
from obspy import read, Stream
import warnings


class SeismicDataProcessor:
    """
    Processor for seismic waveform data.
    
    This class handles loading, preprocessing, and converting seismic data
    for use with PyTorch models.
    
    Args:
        sampling_rate (float): Target sampling rate in Hz
        window_length (float): Length of time window in seconds
        normalize (bool): Whether to normalize the data
    """
    
    def __init__(self, sampling_rate=100.0, window_length=30.0, normalize=True):
        self.sampling_rate = sampling_rate
        self.window_length = window_length
        self.normalize = normalize
        self.n_samples = int(sampling_rate * window_length)
    
    def load_seismic_data(self, filepath, starttime=None, endtime=None):
        """
        Load seismic data from file using ObsPy.
        
        Args:
            filepath (str): Path to seismic data file (supports multiple formats)
            starttime (UTCDateTime): Optional start time for trimming
            endtime (UTCDateTime): Optional end time for trimming
            
        Returns:
            obspy.Stream: Loaded seismic data stream
        """
        try:
            stream = read(filepath)
            
            if starttime is not None or endtime is not None:
                stream = stream.trim(starttime=starttime, endtime=endtime)
            
            return stream
        except Exception as e:
            raise IOError(f"Error loading seismic data from {filepath}: {str(e)}")
    
    def preprocess_stream(self, stream, freqmin=1.0, freqmax=20.0):
        """
        Preprocess seismic stream with filtering and resampling.
        
        Args:
            stream (obspy.Stream): Input stream
            freqmin (float): Minimum frequency for bandpass filter (Hz)
            freqmax (float): Maximum frequency for bandpass filter (Hz)
            
        Returns:
            obspy.Stream: Preprocessed stream
        """
        stream = stream.copy()
        
        # Detrend and demean
        stream.detrend('linear')
        stream.detrend('demean')
        
        # Resample to target sampling rate
        if stream[0].stats.sampling_rate != self.sampling_rate:
            stream.resample(self.sampling_rate)
        
        # Apply bandpass filter
        stream.filter('bandpass', freqmin=freqmin, freqmax=freqmax, corners=4, zerophase=True)
        
        return stream
    
    def stream_to_array(self, stream, n_components=3):
        """
        Convert ObsPy stream to numpy array.
        
        Args:
            stream (obspy.Stream): Input stream
            n_components (int): Expected number of components
            
        Returns:
            np.ndarray: Array of shape (n_components, n_samples)
        """
        if len(stream) != n_components:
            warnings.warn(
                f"Expected {n_components} components, got {len(stream)}. "
                f"Padding or truncating."
            )
        
        data = np.zeros((n_components, self.n_samples))
        
        for i, trace in enumerate(stream[:n_components]):
            trace_data = trace.data
            
            # Trim or pad to desired length
            if len(trace_data) > self.n_samples:
                data[i] = trace_data[:self.n_samples]
            else:
                data[i, :len(trace_data)] = trace_data
        
        # Normalize if requested
        if self.normalize:
            data = self._normalize(data)
        
        return data
    
    def _normalize(self, data, method='std'):
        """
        Normalize seismic data.
        
        Args:
            data (np.ndarray): Input data
            method (str): Normalization method ('std', 'minmax', or 'maxabs')
            
        Returns:
            np.ndarray: Normalized data
        """
        if method == 'std':
            # Standardize to zero mean and unit variance
            mean = np.mean(data, axis=-1, keepdims=True)
            std = np.std(data, axis=-1, keepdims=True)
            std = np.where(std == 0, 1, std)  # Avoid division by zero
            return (data - mean) / std
        
        elif method == 'minmax':
            # Scale to [0, 1] range
            min_val = np.min(data, axis=-1, keepdims=True)
            max_val = np.max(data, axis=-1, keepdims=True)
            range_val = max_val - min_val
            range_val = np.where(range_val == 0, 1, range_val)
            return (data - min_val) / range_val
        
        elif method == 'maxabs':
            # Scale by maximum absolute value
            max_abs = np.max(np.abs(data), axis=-1, keepdims=True)
            max_abs = np.where(max_abs == 0, 1, max_abs)
            return data / max_abs
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def to_torch(self, data):
        """
        Convert numpy array to PyTorch tensor.
        
        Args:
            data (np.ndarray): Input numpy array
            
        Returns:
            torch.Tensor: PyTorch tensor
        """
        return torch.from_numpy(data).float()


class SeismicDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for seismic data.
    
    Args:
        data_list (list): List of file paths or data arrays
        labels (list): List of corresponding labels
        processor (SeismicDataProcessor): Data processor instance
        transform (callable): Optional transform to apply to data
    """
    
    def __init__(self, data_list, labels=None, processor=None, transform=None):
        self.data_list = data_list
        self.labels = labels
        self.processor = processor or SeismicDataProcessor()
        self.transform = transform
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        """
        Get a single data sample.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (data, label) if labels are provided, otherwise just data
        """
        data_item = self.data_list[idx]
        
        # If data_item is a file path, load it
        if isinstance(data_item, str):
            stream = self.processor.load_seismic_data(data_item)
            stream = self.processor.preprocess_stream(stream)
            data = self.processor.stream_to_array(stream)
        else:
            data = data_item
        
        # Convert to tensor
        data = self.processor.to_torch(data)
        
        # Apply transform if provided
        if self.transform:
            data = self.transform(data)
        
        # Return with or without label
        if self.labels is not None:
            label = self.labels[idx]
            return data, label
        else:
            return data
