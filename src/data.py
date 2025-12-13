"""
Data processing utilities for seismic data using ObsPy and SeisBench.
"""

import numpy as np
import torch
from obspy import read, Stream, UTCDateTime
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
    
    def preprocess_stream(self, stream, freqmin=1.0, freqmax=20.0, **kwargs):
        """
        Preprocess seismic stream following ObsPy conventions.
        
        This follows the typical ObsPy preprocessing workflow:
        1. Remove instrument response (if response available)
        2. Detrend (remove linear trend)
        3. Demean (remove mean)
        4. Taper edges (cosine taper)
        5. Filter (bandpass)
        6. Resample (if needed)
        
        Args:
            stream (obspy.Stream): Input stream
            freqmin (float): Minimum frequency for bandpass filter (Hz)
            freqmax (float): Maximum frequency for bandpass filter (Hz)
            **kwargs: Additional arguments for filtering and processing
            
        Returns:
            obspy.Stream: Preprocessed stream
        """
        stream = stream.copy()
        
        # Remove response if inventory is provided
        inventory = kwargs.get('inventory', None)
        if inventory is not None:
            try:
                stream.remove_response(inventory=inventory, output='VEL')
            except Exception as e:
                warnings.warn(f"Could not remove response: {str(e)}")
        
        # Detrend: remove linear trend first, then remove mean
        stream.detrend('linear')
        stream.detrend('demean')
        
        # Taper the edges to avoid filtering artifacts
        # Use 5% taper on each end (ObsPy standard)
        taper_fraction = kwargs.get('taper_fraction', 0.05)
        stream.taper(max_percentage=taper_fraction, type='cosine')
        
        # Apply bandpass filter BEFORE resampling (to avoid aliasing)
        # Use zerophase=True for no phase shift (ObsPy recommendation)
        corners = kwargs.get('corners', 4)
        stream.filter('bandpass', freqmin=freqmin, freqmax=freqmax, 
                     corners=corners, zerophase=True)
        
        # Resample to target sampling rate if needed
        # Check if any trace needs resampling
        needs_resampling = False
        for trace in stream:
            if abs(trace.stats.sampling_rate - self.sampling_rate) > 0.01:
                needs_resampling = True
                break
        
        if needs_resampling:
            # Check if we can use decimation (all traces have same sampling rate)
            sampling_rates = [tr.stats.sampling_rate for tr in stream]
            if len(set(sampling_rates)) == 1:  # All traces have same sampling rate
                factor = sampling_rates[0] / self.sampling_rate
                if factor > 1 and abs(factor - round(factor)) < 0.01:
                    # Use decimation for integer factors
                    stream.decimate(int(round(factor)), no_filter=True)
                else:
                    # Use resample for non-integer factors or upsampling
                    stream.resample(self.sampling_rate)
            else:
                # Different sampling rates, resample each trace individually
                stream.resample(self.sampling_rate)
        
        return stream
    
    def stream_to_array(self, stream, n_components=3, component_order=None):
        """
        Convert ObsPy stream to numpy array following ObsPy conventions.
        
        This method properly handles:
        - Stream sorting by channel code (Z, N, E order)
        - Trace metadata preservation in returned info
        - Proper handling of gaps and overlaps
        
        Args:
            stream (obspy.Stream): Input stream
            n_components (int): Expected number of components
            component_order (list): Optional list specifying component order 
                                   (e.g., ['Z', 'N', 'E'] or ['HHZ', 'HHN', 'HHE'])
            
        Returns:
            np.ndarray: Array of shape (n_components, n_samples)
        """
        # Merge stream to handle any gaps/overlaps (ObsPy best practice)
        stream = stream.copy()
        stream.merge(fill_value=0)
        
        # Sort traces by channel code for consistent ordering (ObsPy convention)
        # Standard order: Z (vertical), N (north), E (east)
        if component_order is None:
            # Try to automatically determine component order from channel codes
            stream = self._sort_stream_by_component(stream)
        else:
            # Sort according to user-specified order
            stream = self._sort_stream_by_order(stream, component_order)
        
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
    
    def _sort_stream_by_component(self, stream):
        """
        Sort stream by component following ObsPy/seismology conventions.
        
        Standard order: Z (vertical), N (north), E (east)
        Common channel naming: HHZ, HHN, HHE or BHZ, BHN, BHE
        
        Args:
            stream (obspy.Stream): Input stream
            
        Returns:
            obspy.Stream: Sorted stream
        """
        # Component priority (last character of channel code)
        component_priority = {'Z': 0, 'N': 1, 'E': 2, '1': 0, '2': 1, '3': 2}
        
        def get_component_priority(trace):
            channel = trace.stats.channel
            # Get last character of channel code
            component = channel[-1] if channel else 'Z'
            return component_priority.get(component, 99)
        
        # Sort traces by component priority
        sorted_traces = sorted(stream.traces, key=get_component_priority)
        
        # Create new stream with sorted traces
        sorted_stream = Stream(traces=sorted_traces)
        return sorted_stream
    
    def _sort_stream_by_order(self, stream, component_order):
        """
        Sort stream according to specified component order.
        
        Args:
            stream (obspy.Stream): Input stream
            component_order (list): List of channel codes or components
            
        Returns:
            obspy.Stream: Sorted stream
        """
        sorted_traces = []
        for comp in component_order:
            for trace in stream:
                if trace.stats.channel.endswith(comp) or trace.stats.channel == comp:
                    sorted_traces.append(trace)
                    break
        
        return Stream(traces=sorted_traces)
    
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
