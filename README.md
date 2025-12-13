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
