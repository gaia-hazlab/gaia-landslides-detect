"""
SeisBench model integration for seismic event detection.

This module provides simple wrappers for SeisBench pre-trained models.
"""

import torch
import numpy as np
from obspy import Stream
from typing import Dict, List, Optional, Tuple
import warnings
from tqdm import tqdm

try:
    import seisbench.models as sbm
    SEISBENCH_AVAILABLE = True
except ImportError:
    SEISBENCH_AVAILABLE = False
    sbm = None
    warnings.warn("SeisBench not available. Install with: pip install seisbench")


class SeismicDetector:
    """
    Simple wrapper for SeisBench models.
    
    The model is already loaded via sbm.ModelName.from_pretrained("version")
    This class just provides a consistent interface.
    
    Args:
        model: Pre-loaded SeisBench model
        model_name (str): Model name for reference
        version (str): Version string for reference
        device (str): Device ('cpu' or 'cuda')
    """
    
    def __init__(
        self,
        model,
        model_name: str = 'phasenet',
        version: str = 'stead',
        device: str = 'cpu'
    ):
        self.model = model
        self.model_name = model_name.lower()
        self.version = version
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def annotate(
        self,
        stream: Stream,
        batch_size: int = 32,
        overlap: int = 0,
        **kwargs
    ) -> Stream:
        """
        Annotate stream with model predictions.
        
        This follows the SeisBench annotate() workflow, which adds prediction
        traces to the input stream.
        
        Args:
            stream (Stream): Input ObsPy stream
            batch_size (int): Batch size for processing
            overlap (int): Overlap between windows in samples
            **kwargs: Additional arguments for model.annotate()
            
        Returns:
            Stream: Annotated stream with prediction traces
        """
        # Use SeisBench's built-in annotate method
        annotated_stream = self.model.annotate(
            stream,
            batch_size=batch_size,
            overlap=overlap,
            **kwargs
        )
        
        return annotated_stream
    
    def classify(
        self,
        stream: Stream,
        **kwargs
    ):
        """
        Get deterministic picks using SeisBench classify() method.
        
        This returns a ClassifyOutput object with deterministic picks,
        following the SeisBench workflow.
        
        Args:
            stream (Stream): Input ObsPy stream
            **kwargs: Additional arguments for model.classify()
            
        Returns:
            ClassifyOutput or Dict: SeisBench ClassifyOutput with picks,
                                   or dict of probabilities if classify not available
                                   
        Example:
            >>> picks_output = detector.classify(stream)
            >>> for pick in picks_output.picks:
            ...     print(f"{pick.phase} at {pick.peak_time} (prob: {pick.peak_value})")
        """
        # Use SeisBench's classify method
        if hasattr(self.model, 'classify'):
            return self.model.classify(stream, **kwargs)
        else:
            warnings.warn(
                f"{self.model_name} does not have classify() method. "
                f"Use annotate() to get probability traces instead."
            )
            return self._extract_predictions(self.annotate(stream, **kwargs))
    
    def _extract_predictions(self, annotated_stream: Stream) -> Dict[str, np.ndarray]:
        """
        Extract prediction arrays from annotated stream.
        
        SeisBench models output channels like:
        - PhaseNet: "PhaseNet_P", "PhaseNet_S", "PhaseNet_N" (noise)
        - EQTransformer: "EQTransformer_P", "EQTransformer_S", "EQTransformer_D" (detection)
        - GPD: "GPD_D" (detection)
        
        Returns:
            Dict[str, np.ndarray]: Dictionary mapping phase names to probability arrays
        """
        predictions = {}
        
        # Map of model prefixes to phase names
        phase_mapping = {
            'P': 'P',
            'S': 'S', 
            'N': 'Noise',
            'D': 'Detection'
        }
        
        # Get all traces from annotated stream
        for trace in annotated_stream:
            channel = trace.stats.channel
            
            # Extract phase type from channel name (e.g., "PhaseNet_P" -> "P")
            if '_' in channel:
                phase_code = channel.split('_')[-1]
                phase_name = phase_mapping.get(phase_code, phase_code)
                
                # Use the trace data directly
                predictions[phase_name] = trace.data
        
        return predictions


class QuakeScopeDetector(SeismicDetector):
    """
    Detector using QuakeScope pre-trained models.
    
    QuakeScope provides models trained on specific regional datasets.
    This class handles loading pre-downloaded weights from QuakeScope.
    Use download_quakescope_model() to download weights first.
    
    Args:
        model_type (str): Model type ('quakexnet', 'phasenet', etc.)
        version (str): Model version
        device (str): Device to run on
    """
    
    def __init__(
        self,
        model_type: str = 'quakexnet',
        version: str = 'base',
        device: str = 'cpu'
    ):
        self.model_type = model_type.lower()
        
        # Map QuakeScope model names to SeisBench
        if model_type.lower() == 'quakexnet':
            # QuakeXNet is a custom model, need to load appropriately
            super().__init__(
                model_name='phasenet',  # Base architecture
                version=version,
                device=device
            )
        else:
            super().__init__(
                model_name=model_type,
                version=version,
                device=device
            )
    
    def load_quakescope_weights(self, weights_path: str):
        """
        Load custom weights from QuakeScope.
        
        Args:
            weights_path (str): Path to weights file (.pt or .pth)
        """
        checkpoint = torch.load(weights_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        print(f"Loaded QuakeScope weights from {weights_path}")


def download_quakescope_model(
    model_name: str,
    save_dir: str = './models',
    version: str = 'latest'
) -> str:
    """
    Download pre-trained model from QuakeScope.
    
    Args:
        model_name (str): Name of the model
        save_dir (str): Directory to save the model
        version (str): Model version
        
    Returns:
        str: Path to downloaded model
    """
    import os
    import urllib.request
    from pathlib import Path
    
    # Create save directory
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # QuakeScope model URLs (these would be actual URLs in production)
    base_url = "https://github.com/quakescope/models/releases/download"
    
    model_urls = {
        'quakexnet_base': f"{base_url}/v1.0/quakexnet_base.pth",
        'quakexnet_v3': f"{base_url}/v3.0/quakexnet_v3.pth",
        'phasenet_regional': f"{base_url}/v1.0/phasenet_regional.pth",
    }
    
    model_key = f"{model_name}_{version}".lower()
    
    if model_key not in model_urls:
        available = list(model_urls.keys())
        raise ValueError(
            f"Model '{model_key}' not found. Available models: {available}"
        )
    
    url = model_urls[model_key]
    save_path = os.path.join(save_dir, f"{model_key}.pth")
    
    if not os.path.exists(save_path):
        print(f"Downloading {model_name} from {url}...")
        try:
            urllib.request.urlretrieve(url, save_path)
            print(f"Model saved to {save_path}")
        except Exception as e:
            warnings.warn(f"Failed to download model: {e}")
            # In practice, we might want to use a local cache or handle this differently
            raise
    else:
        print(f"Model already exists at {save_path}")
    
    return save_path


def create_detector(
    model_name: str = 'phasenet',
    version: str = 'stead',
    device: str = 'auto'
) -> SeismicDetector:
    """
    Create a seismic phase picker/detector.
    
    Follows SeisBench workflow: 
        pn_model = sbm.PhaseNet.from_pretrained("stead")
    
    Args:
        model_name (str): 'phasenet', 'eqtransformer', or 'gpd'
        version (str): Version/training dataset (default: 'stead')
            PhaseNet: 'original', 'ethz', 'instance', 'scedc', 'stead', 
                      'geofon', 'neic'
            EQTransformer: 'original', 'ethz', 'instance', 'scedc', 
                           'stead', 'geofon'
            GPD: 'original', 'ethz', 'scedc', 'stead', 'geofon', 'neic'
        device (str): 'auto', 'cpu', or 'cuda'
        
    Returns:
        SeismicDetector: Detector ready for annotate() or classify()
        
    Example:
        >>> detector = create_detector('phasenet', 'stead')
        >>> annotated = detector.annotate(stream)
        >>> picks = detector.model.classify(stream)
    """
    if not SEISBENCH_AVAILABLE:
        raise ImportError("SeisBench required. Install: pip install seisbench")
    
    # Determine device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Loading {model_name} (version: {version}) on {device}...")
    
    # Load model directly using SeisBench
    model_map = {
        'phasenet': sbm.PhaseNet,
        'eqtransformer': sbm.EQTransformer,
        'gpd': sbm.GPD,
    }
    
    if model_name.lower() not in model_map:
        raise ValueError(
            f"Model '{model_name}' not supported. "
            f"Available: {list(model_map.keys())}"
        )
    
    model_class = model_map[model_name.lower()]
    
    # This is the key SeisBench pattern
    model = model_class.from_pretrained(version)
    
    # Wrap in our detector class
    detector = SeismicDetector(
        model=model,
        model_name=model_name,
        version=version,
        device=device
    )
    
    print(f"✓ Detector ready")
    return detector


def convert_picks_to_detections(
    picks_output,
    sampling_rate: float = 100.0,
    stream_starttime=None
) -> Dict[str, List[Dict]]:
    """
    Convert SeisBench ClassifyOutput picks to detection format.
    
    Args:
        picks_output: SeisBench ClassifyOutput object from model.classify()
        sampling_rate (float): Sampling rate in Hz
        stream_starttime: ObsPy UTCDateTime of stream start (optional)
        
    Returns:
        Dict[str, List[Dict]]: Dictionary of picks by phase (P, S, etc.)
        
    Example:
        >>> picks_output = detector.model.classify(stream)
        >>> pick_detections = convert_picks_to_detections(picks_output, 
        ...                                                stream[0].stats.sampling_rate,
        ...                                                stream[0].stats.starttime)
        >>> print(pick_detections['P'])  # All P picks
    """
    detections = {}
    
    if not hasattr(picks_output, 'picks'):
        warnings.warn("Input does not have 'picks' attribute. Expected ClassifyOutput.")
        return detections
    
    # Group picks by phase
    for pick in picks_output.picks:
        phase = pick.phase
        
        if phase not in detections:
            detections[phase] = []
        
        # Get pick time in seconds
        if hasattr(pick.peak_time, 'timestamp'):
            # It's a UTCDateTime object
            if stream_starttime is not None:
                pick_time_sec = float(pick.peak_time - stream_starttime)
            else:
                pick_time_sec = pick.peak_time.timestamp
        else:
            # It's already a float/number
            pick_time_sec = float(pick.peak_time)
        
        # Convert to detection format compatible with our workflow
        pick_dict = {
            'phase': phase,
            'pick_time': pick_time_sec,
            'peak_time': pick_time_sec,
            'max_prob': float(pick.peak_value),
            'pick_prob': float(pick.peak_value),
            'start': int(pick_time_sec * sampling_rate),
            'end': int(pick_time_sec * sampling_rate),
            'peak_idx': int(pick_time_sec * sampling_rate),
            'trace_id': pick.trace_id if hasattr(pick, 'trace_id') else None
        }
        
        detections[phase].append(pick_dict)
    
    return detections



class SeismicClassifier:
    """
    Wrapper for QuakeXNet model to perform seismic event classification.
    
    Classifies waveform windows into event types: earthquake (eq), explosion (px),
    noise (no), and surface event (su).
    
    Args:
        model: QuakeXNet model instance
        device (str): Device to run on ('cpu' or 'cuda')
        labels (List[str]): Class labels (default: ['eq', 'px', 'no', 'su'])
    """
    
    def __init__(
        self,
        model,
        device: str = 'cpu',
        labels: Optional[List[str]] = None
    ):
        self.model = model
        self.device = device
        self.labels = labels or ['eq', 'px', 'no', 'su']
        self.model.to(device)
        self.model.eval()
    
    def classify_windows(
        self,
        windows: List[np.ndarray],
        batch_size: int = 12
    ) -> List[Tuple[np.ndarray, str]]:
        """
        Classify waveform windows in batches.
        
        Args:
            windows (List[np.ndarray]): List of waveform windows (n_components, n_samples)
            batch_size (int): Batch size for processing (default: 12)
            
        Returns:
            List[Tuple[np.ndarray, str]]: List of (class_probabilities, predicted_label)
        """
        results = []
        n_windows = len(windows)
        
        with torch.no_grad():
            # Process in batches with progress bar
            for i in tqdm(range(0, n_windows, batch_size), desc="Classifying windows"):
                batch_windows = windows[i:i + batch_size]
                
                # Stack into tensor (batch_size, n_components, n_samples)
                batch_tensor = torch.tensor(
                    np.array(batch_windows), 
                    dtype=torch.float32
                ).to(self.device)
                
                # Run model inference
                # QuakeXNet expects input and handles preprocessing internally
                outputs = self.model(batch_tensor)
                
                # Get probabilities (softmax already applied in annotate_batch_post)
                if outputs.dim() == 2:
                    # Already probabilities
                    probs = outputs.cpu().numpy()
                else:
                    # Apply softmax if needed
                    probs = torch.softmax(outputs, dim=-1).cpu().numpy()
                
                # Get predicted labels
                for j in range(len(batch_windows)):
                    class_probs = probs[j]
                    predicted_idx = np.argmax(class_probs)
                    predicted_label = self.labels[predicted_idx]
                    results.append((class_probs, predicted_label))
        
        return results
    
    def load_weights(self, weights_path: str):
        """
        Load model weights from file.
        
        Args:
            weights_path (str): Path to weights file (.pt or .pth)
        """
        checkpoint = torch.load(weights_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        print(f"Loaded weights from {weights_path}")


def download_model_weights(
    model_name: str,
    version: str = 'base',
    custom_url: Optional[str] = None,
    save_dir: Optional[str] = None
) -> str:
    """
    Download model weights with flexible source options.
    
    Args:
        model_name (str): Model name (e.g., 'quakexnet', 'phasenet')
        version (str): Model version (e.g., 'base', 'v3')
        custom_url (str): Optional custom URL for weights
        save_dir (str): Optional custom save directory (default: ~/.cache/gaia-landslides/models/)
        
    Returns:
        str: Path to downloaded weights file
    """
    import os
    import urllib.request
    from pathlib import Path
    
    # Determine save directory
    if save_dir is None:
        save_dir = os.path.expanduser('~/.cache/gaia-landslides/models')
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Determine filename and URL
    filename = f"{model_name}_{version}.pt"
    save_path = os.path.join(save_dir, filename)
    
    # Check if file already exists
    if os.path.exists(save_path):
        print(f"Weights already cached at {save_path}")
        return save_path
    
    # Determine download URL
    if custom_url:
        url = custom_url
    else:
        # Default QuakeScope URL pattern
        base_url = "https://github.com/SeisSCOPED/QuakeScope/raw/main/sb_catalog/models"
        url = f"{base_url}/{model_name}/{version}.pt.v1"
    
    print(f"Downloading {model_name} weights from {url}...")
    print(f"Saving to {save_path}...")
    
    try:
        # Download with progress
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 / total_size) if total_size > 0 else 0
            print(f"\rProgress: {percent:.1f}%", end='')
        
        urllib.request.urlretrieve(url, save_path, reporthook=report_progress)
        print(f"\n✓ Download complete: {save_path}")
        
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        # Clean up partial download
        if os.path.exists(save_path):
            os.remove(save_path)
        raise RuntimeError(
            f"Failed to download weights from {url}. "
            f"Please check the URL or provide a local weights file."
        )
    
    return save_path


def create_classifier(
    model_name: str = 'quakexnet',
    version: str = 'base',
    device: str = 'auto',
    weights_path: Optional[str] = None,
    auto_download: bool = True
) -> SeismicClassifier:
    """
    Factory function to create a seismic event classifier.
    
    Args:
        model_name (str): Model name (default: 'quakexnet')
        version (str): Model version (default: 'base')
        device (str): Device ('auto', 'cpu', or 'cuda')
        weights_path (str): Optional path to custom weights
        auto_download (bool): Automatically download weights if not found
        
    Returns:
        SeismicClassifier: Configured classifier instance
    """
    from src.models import QuakeXNet
    
    # Determine device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Creating {model_name} classifier on {device}...")
    
    # Initialize model
    if model_name.lower() == 'quakexnet':
        model = QuakeXNet(
            sampling_rate=50,
            classes=4,
            labels=['eq', 'px', 'no', 'su']
        )
    else:
        raise ValueError(f"Model '{model_name}' not supported for classification")
    
    # Create classifier wrapper
    classifier = SeismicClassifier(model=model, device=device)
    
    # Load weights
    if weights_path:
        # Use provided weights path
        print(f"Loading weights from {weights_path}...")
        classifier.load_weights(weights_path)
    elif auto_download:
        # Download weights
        try:
            weights_path = download_model_weights(
                model_name=model_name,
                version=version
            )
            classifier.load_weights(weights_path)
        except Exception as e:
            warnings.warn(
                f"Could not download weights: {e}. "
                f"Classifier initialized with random weights."
            )
    else:
        warnings.warn("No weights loaded. Classifier initialized with random weights.")
    
    print(f"✓ Classifier ready")
    return classifier

