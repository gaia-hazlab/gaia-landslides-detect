# Two-Step Seismic Workflow Implementation Summary

## Completed Implementation

### 1. Core Functions in `src/detect.py`

#### Window Extraction Functions
- **`extract_fixed_windows()`**: Extracts 100-second sliding windows with configurable stride (default 50s)
  - Handles resampling to 50Hz
  - Zero-padding at boundaries
  - Returns windows, center times, and indices

- **`extract_event_centered_windows()`**: Extracts windows centered on detected events/picks
  - 100-second windows centered on event times
  - Boundary padding support
  - Useful for event-focused classification

- **`prepare_classification_batch()`**: Prepares windows for batch processing
  - Stacks windows into PyTorch tensors
  - Default batch size of 12

#### Classification Function
- **`classify_waveform_windows()`**: Hybrid classification approach
  - Runs both sliding window and event-centered classification
  - Progress bar for batch processing (tqdm)
  - Returns comprehensive results with class probabilities

#### Merging Function
- **`merge_picks_and_classifications()`**: Intelligent merging of picks and classifications
  - ±10 second temporal tolerance (configurable)
  - Closest match resolution when multiple overlaps
  - Preserves unmatched picks and classifications
  - Returns pandas DataFrame with match types: 'matched', 'pick_only', 'class_only'

### 2. Classifier Infrastructure in `src/seisbench_models.py`

#### SeismicClassifier Class
- **`classify_windows()`**: Batch classification with progress tracking
  - Processes windows in batches of 12
  - Returns class probabilities and predicted labels
  - Handles QuakeXNet preprocessing internally

- **`load_weights()`**: Flexible weight loading from files or URLs

#### Weight Management
- **`download_model_weights()`**: Automatic weight download
  - Downloads from QuakeScope GitHub repository
  - Caches to `~/.cache/gaia-landslides/models/`
  - Progress reporting
  - Skips if already cached

#### Factory Function
- **`create_classifier()`**: Easy classifier instantiation
  - Automatic device selection (CPU/CUDA)
  - Auto-downloads weights from QuakeScope
  - Supports manual weight paths
  - Initializes QuakeXNet model

### 3. Updated Notebook: `notebooks/seisbench_detection.ipynb`

#### Restructured Workflow
- **Section A**: Phase Picking (PhaseNet/EQTransformer)
  - Load picker model
  - Run inference on waveforms
  - Detect P/S picks

- **Section B**: Event Classification (Parallel)
  - Load QuakeXNet classifier
  - Run hybrid classification (sliding + event-centered)
  - Get event type predictions (eq, px, no, su)

- **Section C**: Merging Results
  - Match picks with classifications
  - Create comprehensive event catalog
  - Export merged results

- **Section D**: Analysis & Visualization
  - Summary statistics by match type
  - Event class distribution
  - Filtered exports (e.g., surface events only)

## Configuration Parameters

### Picker Settings
- `picker_model`: 'phasenet', 'eqtransformer', 'gpd'
- `picker_threshold`: 0.5 (detection threshold)
- `min_duration`: 10 samples
- `merge_distance`: 50 samples

### Classifier Settings
- `classifier_model`: 'quakexnet'
- `classifier_version`: 'base' or 'v3'
- `window_duration`: 100 seconds (5000 samples @ 50Hz)
- `stride`: 50 seconds (for sliding windows)
- `batch_size`: 12
- `include_sliding`: True (sliding windows)
- `include_event_centered`: True (event-centered windows)

### Merging Settings
- `time_tolerance`: 10.0 seconds (±10s matching window)

## Output DataFrame Schema

Merged results DataFrame includes:
- `pick_time`: Time of phase pick (or None)
- `pick_phase`: Phase type (P, S, or None)
- `pick_prob`: Pick probability
- `class_label`: Event class (eq, px, no, su, or None)
- `class_prob`: Maximum class probability
- `class_prob_eq`: Earthquake probability
- `class_prob_px`: Explosion probability
- `class_prob_no`: Noise probability
- `class_prob_su`: Surface event probability
- `match_type`: 'matched', 'pick_only', or 'class_only'
- `time_diff`: Time difference between pick and classification (seconds)
- `window_type`: 'sliding' or 'event_centered'

## Key Features

1. **Parallel Processing**: Picks and classifications run independently
2. **Hybrid Classification**: Both sliding and event-centered approaches
3. **Intelligent Merging**: Preserves all data while identifying matches
4. **Flexible Weight Management**: Auto-download from QuakeScope with caching
5. **Progress Tracking**: tqdm progress bars for long-running operations
6. **Complete Event Catalog**: Includes matched, pick-only, and class-only events

## Usage Example

```python
from src.seisbench_models import create_detector, create_classifier
from src.detect import classify_waveform_windows, merge_picks_and_classifications

# Load models
picker = create_detector('phasenet')
classifier = create_classifier('quakexnet', version='base')

# Run picker
annotated_stream = picker.annotate(stream)
pick_probabilities = picker._extract_predictions(annotated_stream)
pick_detections = multi_class_detection(pick_probabilities)

# Run classifier (parallel)
classifications = classify_waveform_windows(
    stream, classifier, pick_detections
)

# Merge results
df_merged = merge_picks_and_classifications(
    picker_results, classifications, time_tolerance=10.0
)
```

## Next Steps for Users

1. Install required dependencies: `pip install tqdm pandas`
2. Run the updated notebook: `notebooks/seisbench_detection.ipynb`
3. Weights will auto-download from QuakeScope on first run
4. Adjust parameters based on your data characteristics
5. Filter results by `match_type` or `class_label` for specific analyses
