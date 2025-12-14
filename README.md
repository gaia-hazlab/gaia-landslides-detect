# GAIA Landslides Detection

A PyTorch-based framework for deploying deep learning models for seismic data analysis and landslide detection, with integrated SeisBench workflow for event detection.

## Features

- PyTorch-based neural network models for seismic data classification
- Integration with ObsPy for seismic data processing following seismology conventions
- SeisBench workflow for automated event detection (PhaseNet, EQTransformer, GPD)
- QuakeScope model support with pre-trained weights
- Interactive visualization tools for detection quality control
- Jupyter notebook examples for complete workflows

## Repository Structure

```
gaia-landslides-detect/
├── src/                           # Source code
│   ├── __init__.py               # Package initialization
│   ├── models.py                 # PyTorch model definitions
│   ├── data.py                   # ObsPy-compliant data processing
│   ├── utils.py                  # Visualization and helper functions
│   ├── detect.py                 # Event detection utilities
│   ├── seisbench_models.py       # SeisBench model integration
│   └── interactive_plots.py      # Interactive QC visualization
├── notebooks/                     # Jupyter notebooks
│   ├── example_analysis.ipynb    # Basic analysis workflow
│   └── seisbench_detection.ipynb # SeisBench detection workflow
├── plots/                         # Generated plots and visualizations
├── data/                          # Seismic data
│   ├── raw/                      # Raw input data
│   └── processed/                # Processed datasets
├── tests/                         # Unit tests
├── config.yaml                    # Configuration file
├── requirements.txt               # Python dependencies
├── environment.yml                # Conda environment
├── setup.py                       # Package setup file
├── pyproject.toml                 # Project configuration
└── README.md                      # This file
```

## Installation

### Option 1: Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/gaia-hazlab/gaia-landslides-detect.git
cd gaia-landslides-detect

# Create and activate conda environment
conda env create -f environment.yml
conda activate gaia-hazlab

# Install the package in development mode
pip install -e .
```

### Option 2: Using pip

```bash
# Clone the repository
git clone https://github.com/gaia-hazlab/gaia-landslides-detect.git
cd gaia-landslides-detect

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Quick Start

### SeisBench Detection Workflow (Recommended)

The fastest way to get started with seismic event detection:

```python
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from src.seisbench_models import create_detector
from src.detect import multi_class_detection
from src.interactive_plots import plot_detection_results

# 1. Load pre-trained SeisBench model
detector = create_detector(
    model_name='phasenet',  # or 'eqtransformer', 'gpd'
    device='auto'
)

# 2. Download seismic data
client = Client("IRIS")
stream = client.get_waveforms(
    network="UW", station="RATT", channel="HH*", location="*",
    starttime=UTCDateTime() - 3600,
    endtime=UTCDateTime()
)

# 3. Run detection
annotated_stream = detector.annotate(stream)
probabilities = detector._extract_predictions(annotated_stream)

# 4. Detect events
events = multi_class_detection(probabilities, threshold=0.5)

# 5. Visualize results
fig, axes = plot_detection_results(stream, probabilities, events)
```

See `notebooks/seisbench_detection.ipynb` for the complete interactive workflow.

### Custom Model Training

```python
import torch
from src.models import SeismicCNN, load_model, save_model
from src.data import SeismicDataProcessor
from src.utils import plot_seismogram, get_device

# Initialize device
device = get_device()

# Create a model
model = SeismicCNN(input_channels=3, num_classes=2)
model = model.to(device)

# Initialize data processor
processor = SeismicDataProcessor(
    sampling_rate=100.0,
    window_length=30.0,
    normalize=True
)

# Load and process seismic data
stream = processor.load_seismic_data('path/to/seismic_file.mseed')
stream = processor.preprocess_stream(stream, freqmin=1.0, freqmax=20.0)
data = processor.stream_to_array(stream)

# Make predictions
data_tensor = processor.to_torch(data).unsqueeze(0).to(device)
model.eval()
with torch.no_grad():
    output = model(data_tensor)
    probabilities = torch.softmax(output, dim=1)

print(f"Predictions: {probabilities}")
```

### Using the Jupyter Notebooks

#### SeisBench Detection (Recommended)
1. Start Jupyter:
```bash
jupyter notebook
```

2. Open `notebooks/seisbench_detection.ipynb`

3. Follow the workflow:
   - Load SeisBench models (PhaseNet, EQTransformer, or custom QuakeScope models)
   - Download data from IRIS/FDSN
   - Run automated detection
   - Interactive QC visualization
   - Export results to CSV

#### Basic Analysis
Open `notebooks/example_analysis.ipynb` for:
   - Creating synthetic seismograms
   - Visualizing seismic data
   - Training custom models
   - Saving and loading checkpoints

## Dependencies

Core dependencies:
- PyTorch (>=2.0.0): Deep learning framework
- ObsPy (>=1.4.0): Seismic data processing
- SeisBench (>=0.4.0): Pre-trained seismic models and benchmarks
- NumPy, SciPy, Pandas: Scientific computing and data analysis
- Matplotlib: Visualization

See `requirements.txt` or `environment.yml` for the complete list.

## SeisBench Workflow

This repository implements the SeisBench detection workflow for automated seismic event detection:

### Supported Models
- PhaseNet: P and S wave detection
- EQTransformer: Earthquake detection and phase picking
- GPD: General purpose detector
- QuakeScope Models: Regional models with custom weights

### Detection Pipeline
1. Model Loading: Load pre-trained SeisBench models or custom QuakeScope weights
2. Data Acquisition: Download data from FDSN services (IRIS, NCEDC, etc.)
3. Inference: Run `model.annotate()` to get probability predictions
4. Event Detection: Apply smoothing and threshold-based detection
5. Quality Control: Interactive visualization with toggleable event classes
6. Export: Save results to CSV with event metrics (AUC, probabilities, timing)

### Event Detection Features
- Multi-class detection: Distinguish between earthquakes, explosions, and surface events
- Event metrics: Area under curve, max/mean probabilities, duration
- Smart merging: Merge nearby events with configurable distance
- Non-maximum suppression: Remove overlapping detections
- Interactive QC: Toggle event classes, zoom, and inspect waveforms

See `notebooks/seisbench_detection.ipynb` for the complete workflow implementation.

## Data Format

The framework follows ObsPy conventions for seismic data processing and supports various formats including MiniSEED (.mseed), SAC, SEG-Y, and ASDF.

### Data Processing Workflow
The `SeismicDataProcessor` follows the standard ObsPy preprocessing workflow:

1. Load data using ObsPy's `read()` function with automatic format detection
2. Detrend to remove linear trend and mean
3. Taper using cosine taper to avoid edge effects
4. Filter with bandpass filter
5. Resample to target sampling rate
6. Sort components (Z, N, E order)
7. Merge to handle gaps and overlaps

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. See `CONTRIBUTING.md` for guidelines.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This project builds upon:
- [PyTorch](https://pytorch.org/)
- [ObsPy](https://obspy.org/)
- [SeisBench](https://github.com/seisbench/seisbench)
