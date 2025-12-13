"""
SeisBench model integration for seismic event detection.

This module provides wrappers and utilities for using SeisBench pre-trained models
including PhaseNet, EQTransformer, GPD, and others.
"""

import torch
import numpy as np
from obspy import Stream, UTCDateTime
from typing import Dict, List, Optional, Tuple, Union
import warnings

try:
    import seisbench.models as sbm
    SEISBENCH_AVAILABLE = True
except ImportError:
    SEISBENCH_AVAILABLE = False
    warnings.warn("SeisBench not available. Install with: pip install seisbench")


class SeismicDetector:
    """
    Wrapper for SeisBench models to perform seismic event detection.
    
    Supports models like PhaseNet, EQTransformer, GPD, and custom models
    from QuakeScope or other sources.
    
    Args:
        model_name (str): Name of the SeisBench model (e.g., 'phasenet', 'eqtransformer')
        version (str): Model version (e.g., 'original', 'instance', 'base')
        pretrained (str): Pretrained weights source (e.g., 'original', custom path)
        device (str): Device to run on ('cpu' or 'cuda')
    """
    
    def __init__(
        self,
        model_name: str = 'phasenet',
        version: str = 'original',
        pretrained: Optional[str] = None,
        device: str = 'cpu'
    ):
        if not SEISBENCH_AVAILABLE:
            raise ImportError("SeisBench is required. Install with: pip install seisbench")
        
        self.model_name = model_name.lower()
        self.version = version
        self.device = device
        
        # Load model
        self.model = self._load_model(pretrained)
        self.model.to(device)
        self.model.eval()
        
    def _load_model(self, pretrained: Optional[str] = None):
        """Load SeisBench model."""
        model_map = {
            'phasenet': sbm.PhaseNet,
            'eqtransformer': sbm.EQTransformer,
            'gpd': sbm.GPD,
            'basicphaseae': sbm.BasicPhaseAE,
            'cred': sbm.CRED,
        }
        
        if self.model_name not in model_map:
            raise ValueError(
                f"Model '{self.model_name}' not supported. "
                f"Available models: {list(model_map.keys())}"
            )
        
        model_class = model_map[self.model_name]
        
        # Load pretrained weights
        if pretrained:
            model = model_class.from_pretrained(pretrained, version_str=self.version)
        else:
            model = model_class.from_pretrained(self.version)
        
        return model
    
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
    ) -> Dict[str, np.ndarray]:
        """
        Classify stream and return probability arrays.
        
        Args:
            stream (Stream): Input ObsPy stream
            **kwargs: Additional arguments for model.classify()
            
        Returns:
            Dict[str, np.ndarray]: Dictionary of class probabilities
        """
        # Use SeisBench's classify method if available
        if hasattr(self.model, 'classify'):
            predictions = self.model.classify(stream, **kwargs)
        else:
            # Fallback to annotate and extract predictions
            annotated = self.annotate(stream, **kwargs)
            predictions = self._extract_predictions(annotated)
        
        return predictions
    
    def _extract_predictions(self, annotated_stream: Stream) -> Dict[str, np.ndarray]:
        """Extract prediction arrays from annotated stream."""
        predictions = {}
        
        # Get all unique channel names that contain predictions
        channels = set([tr.stats.channel for tr in annotated_stream])
        
        for channel in channels:
            # Select traces for this channel
            traces = annotated_stream.select(channel=channel)
            if traces:
                # Combine all traces for this channel
                data = np.concatenate([tr.data for tr in traces])
                predictions[channel] = data
        
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
    version: str = 'original',
    device: str = 'auto',
    use_quakescope: bool = False,
    weights_path: Optional[str] = None
) -> SeismicDetector:
    """
    Factory function to create a seismic detector.
    
    Args:
        model_name (str): Model name
        version (str): Model version
        device (str): Device ('auto', 'cpu', or 'cuda')
        use_quakescope (bool): Whether to use QuakeScope models
        weights_path (str): Path to custom weights
        
    Returns:
        SeismicDetector: Configured detector instance
    """
    # Determine device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create detector
    if use_quakescope:
        detector = QuakeScopeDetector(
            model_type=model_name,
            version=version,
            device=device
        )
        
        # Load custom weights if provided
        if weights_path:
            detector.load_quakescope_weights(weights_path)
    else:
        detector = SeismicDetector(
            model_name=model_name,
            version=version,
            device=device
        )
    
    return detector
