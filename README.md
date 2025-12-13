# Gaia Landslides Detection

A PyTorch-based framework for deploying deep learning models for seismic data analysis and landslide detection.

## Features

- 🔥 PyTorch-based neural network models for seismic data classification
- 🌊 Integration with ObsPy for seismic data processing
- 📊 SeisBench compatibility for earthquake detection benchmarks
- 📈 Visualization tools for seismograms and spectrograms
- 🚀 Ready-to-use template for model deployment
- 📓 Jupyter notebook examples for interactive analysis

## Repository Structure

```
gaia-landslides-detect/
├── src/                    # Source code
│   ├── __init__.py        # Package initialization
│   ├── models.py          # PyTorch model definitions
│   ├── data.py            # Data processing utilities
│   └── utils.py           # Visualization and helper functions
├── notebooks/             # Jupyter notebooks
│   └── example_analysis.ipynb  # Example analysis workflow
├── plots/                 # Generated plots and visualizations
├── data/                  # Seismic data files
├── requirements.txt       # Python dependencies
├── setup.py              # Package setup file
├── pyproject.toml        # Project configuration
└── README.md             # This file
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/gaia-hazlab/gaia-landslides-detect.git
cd gaia-landslides-detect
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install the package in development mode

```bash
pip install -e .
```

## Quick Start

### Using the Python API

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

### Using the Jupyter Notebook

1. Start Jupyter:
```bash
jupyter notebook
```

2. Open `notebooks/example_analysis.ipynb`

3. Run the cells to see examples of:
   - Creating synthetic seismograms
   - Visualizing seismic data
   - Training and evaluating models
   - Saving and loading checkpoints

## Dependencies

Core dependencies:
- **PyTorch** (>=2.0.0): Deep learning framework
- **ObsPy** (>=1.4.0): Seismic data processing
- **SeisBench** (>=0.4.0): Seismic benchmark datasets
- **Matplotlib** (>=3.7.0): Visualization
- **NumPy** (>=1.24.0): Numerical computing
- **SciPy** (>=1.10.0): Scientific computing

See `requirements.txt` for the complete list.

## Model Architecture

The default `SeismicCNN` model includes:
- 3 convolutional blocks with batch normalization and max pooling
- Adaptive average pooling for variable-length inputs
- Fully connected classifier with dropout for regularization

The architecture is designed for 1D seismic waveform data and can be easily customized for different tasks.

## Data Format

The framework follows ObsPy conventions for seismic data processing:

### Supported Formats
ObsPy supports various seismic data formats, including:
- MiniSEED (.mseed)
- SAC (Seismic Analysis Code)
- SEG-Y
- ASDF (Adaptable Seismic Data Format)
- And many more

### Data Processing Workflow
The `SeismicDataProcessor` follows the standard ObsPy preprocessing workflow:

1. **Load data**: Uses ObsPy's `read()` function with automatic format detection
2. **Detrend**: Removes linear trend and mean
3. **Taper**: Applies cosine taper to avoid edge effects (5% by default)
4. **Filter**: Applies bandpass filter (before resampling to avoid aliasing)
5. **Resample**: Resamples to target sampling rate using decimation when possible
6. **Sort components**: Automatically sorts traces by component (Z, N, E order)
7. **Merge**: Handles gaps and overlaps using ObsPy's merge function

### Component Ordering
The framework automatically sorts seismic traces following seismology conventions:
- **Z** (or 1): Vertical component
- **N** (or 2): North component  
- **E** (or 3): East component

This ensures consistent data ordering regardless of how traces are stored in the file.

## Usage Examples

### Training a Model

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.models import SeismicCNN
from src.data import SeismicDataset, SeismicDataProcessor
from src.utils import set_seed, get_device

# Set seed for reproducibility
set_seed(42)
device = get_device()

# Create dataset and dataloader
processor = SeismicDataProcessor()
dataset = SeismicDataset(data_list, labels, processor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model
model = SeismicCNN(input_channels=3, num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
model.train()
for epoch in range(num_epochs):
    for data, labels in dataloader:
        data, labels = data.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

### Visualizing Results

```python
from src.utils import plot_seismogram, plot_spectrogram

# Plot seismogram
plot_seismogram(
    data,
    sampling_rate=100.0,
    title="Seismic Event",
    labels=['Z', 'N', 'E'],
    save_path='plots/seismogram.png'
)

# Plot spectrogram
plot_spectrogram(
    data,
    sampling_rate=100.0,
    title="Frequency Analysis",
    save_path='plots/spectrogram.png'
)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{gaia_landslides_detect,
  title = {Gaia Landslides Detection},
  author = {Gaia Hazlab},
  year = {2024},
  url = {https://github.com/gaia-hazlab/gaia-landslides-detect}
}
```

## Contact

For questions or issues, please open an issue on GitHub.

## Acknowledgments

This project builds upon:
- [PyTorch](https://pytorch.org/)
- [ObsPy](https://obspy.org/)
- [SeisBench](https://github.com/seisbench/seisbench)
# GAIA Landslides Detect

This repository bootstraps a research sandbox for detecting landslide-related signals from geophysical and remote-sensing data. It organizes notebooks, reusable Python code, configuration files, and data staging areas so experiments can start quickly and remain reproducible.

## Repository Structure
- `notebooks/`: exploratory analyses and reporting notebooks.
- `src/gaia_hazlab/`: Python source code for data loading, feature extraction, and modeling.
- `configs/`: configuration files for experiments and data processing.
- `data/raw/`: immutable input data from external sources.
- `data/processed/`: cleaned and intermediate datasets generated from pipelines.
- `docs/`: project documentation and design notes.

## Environment Setup
1. Install [Conda](https://docs.conda.io/) or [Mamba](https://mamba.readthedocs.io/).
2. Create and activate the project environment:
   ```bash
   conda env create -f environment.yml
   conda activate gaia-hazlab
   ```
3. Install the repository in editable mode (optional, for local imports):
   ```bash
   pip install -e .
   ```

The environment pins core scientific packages (ObsPy, NumPy, Pandas, Matplotlib, Plotly, scikit-learn) and pulls in seismic and surface-event detection tooling via git (QuakeScope and Surface_Event_Detection).

## Data Directories
- Place raw inputs under `data/raw/` (kept read-only once ingested).
- Write processed artifacts to `data/processed/`.
- The `.gitignore` excludes data folders by default to avoid committing large files. Keep small configuration or schema samples in `configs/`.

## Quickstart
1. Open a new notebook in `notebooks/` and import utilities from `src/gaia_hazlab/` once added.
2. Point scripts to configuration files stored in `configs/` and read raw data from `data/raw/`.
3. Save derived tables or features to `data/processed/` for downstream models.
4. Document workflows and results in `docs/`.

With this scaffold in place, you can focus on building detection pipelines and analyses without repeatedly recreating project scaffolding.
