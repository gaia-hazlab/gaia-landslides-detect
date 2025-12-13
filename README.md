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

The framework supports various seismic data formats through ObsPy, including:
- MiniSEED (.mseed)
- SAC (Seismic Analysis Code)
- SEG-Y
- And many more

Data should be organized as multi-channel time series (e.g., 3-component seismograms: Z, N, E).

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