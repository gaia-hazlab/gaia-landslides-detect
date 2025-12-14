# Quick Reference Guide: Two-Step Workflow

## Installation

```bash
# Install required packages
pip install tqdm pandas

# Or install all dependencies
pip install -r requirements.txt
```

## Basic Usage

### 1. Load Models

```python
from src.seisbench_models import create_detector, create_classifier

# Load phase picker (following SeisBench workflow)
# Available versions: 'original', 'ethz', 'instance', 'scedc', 'stead', 'geofon', 'neic'
picker = create_detector(
    model_name='phasenet',  # or 'eqtransformer', 'gpd'
    version='stead',        # STEAD is a comprehensive training dataset
    device='auto'
)

# Load event classifier
classifier = create_classifier(
    model_name='quakexnet',
    version='base',  # or 'v3'
    device='auto',
    auto_download=True  # Downloads weights from QuakeScope
)
```

### 2. Run Phase Picking

**Method A: Using annotate() for probability traces (recommended for continuous data)**

```python
from src.detect import multi_class_detection

# Run picker inference - returns probability traces
annotated_stream = picker.annotate(stream, batch_size=32)
pick_probabilities = picker._extract_predictions(annotated_stream)

# Detect picks from probabilities
pick_detections = multi_class_detection(
    pick_probabilities,
    threshold=0.5,
    min_duration=10,
    merge_distance=50
)
```

**Method B: Using classify() for deterministic picks (SeisBench native)**

```python
from src.seisbench_models import convert_picks_to_detections

# Get deterministic picks directly
picks_output = picker.model.classify(stream)

# Convert to our workflow format
pick_detections = convert_picks_to_detections(
    picks_output,
    sampling_rate=stream[0].stats.sampling_rate
)

# Access picks
for pick in picks_output.picks:
    print(f"{pick.phase} at {pick.peak_time}: {pick.peak_value:.2f}")
```

### 3. Run Event Classification

```python
from src.detect import classify_waveform_windows

# Run hybrid classification
classifications = classify_waveform_windows(
    stream=stream,
    classifier=classifier,
    picker_detections=pick_detections,  # Optional, for event-centered windows
    window_duration=100.0,  # seconds
    stride=50.0,  # seconds
    batch_size=12,
    include_sliding=True,
    include_event_centered=True
)
```

### 4. Merge Results

```python
from src.detect import merge_picks_and_classifications

# Prepare picker results
sampling_rate = stream[0].stats.sampling_rate
picker_results = []
for phase_name, phase_picks in pick_detections.items():
    for pick in phase_picks:
        pick['phase'] = phase_name
        picker_results.append(pick)

# Merge
df_merged = merge_picks_and_classifications(
    picker_results=picker_results,
    classification_results=classifications,
    time_tolerance=10.0,  # ±10 seconds
    sampling_rate=sampling_rate
)
```

### 5. Filter and Analyze

```python
from src.detect import filter_noise_events, get_event_type_counts

# Remove noise classifications
df_clean = filter_noise_events(df_merged, remove_noise=True)

# Get event counts
counts = get_event_type_counts(df_merged)
print(counts)

# Filter by match type
matched_only = df_merged[df_merged['match_type'] == 'matched']
picks_only = df_merged[df_merged['match_type'] == 'pick_only']
class_only = df_merged[df_merged['match_type'] == 'class_only']

# Filter by event type
earthquakes = df_merged[df_merged['class_label'] == 'eq']
explosions = df_merged[df_merged['class_label'] == 'px']
surface_events = df_merged[df_merged['class_label'] == 'su']
noise = df_merged[df_merged['class_label'] == 'no']
```

## DataFrame Columns

### Merged Results DataFrame

| Column | Type | Description |
|--------|------|-------------|
| `pick_time` | float | Time of phase pick (seconds from start) |
| `pick_phase` | str | Phase type (P, S, etc.) or None |
| `pick_prob` | float | Pick probability (0-1) |
| `class_label` | str | Event class (eq/px/no/su) or None |
| `class_prob` | float | Maximum class probability |
| `class_prob_eq` | float | Earthquake probability |
| `class_prob_px` | float | Explosion probability |
| `class_prob_no` | float | Noise probability |
| `class_prob_su` | float | Surface event probability |
| `match_type` | str | 'matched', 'pick_only', or 'class_only' |
| `time_diff` | float | Time difference for matched events (seconds) |
| `window_type` | str | 'sliding' or 'event_centered' |

## Event Classes

- **eq**: Earthquake
- **px**: Explosion
- **no**: Noise
- **su**: Surface event (landslide, rockfall, etc.)

## Match Types

- **matched**: Event has both a pick and classification
- **pick_only**: Picker detected event, but no classification match
- **class_only**: Classifier detected event, but picker missed it

## Configuration Tips

### Model Selection

**PhaseNet Versions:**
- `original`: Original PhaseNet training
- `ethz`: Trained on ETH Zurich data
- `instance`: Instance-based learning
- `scedc`: Southern California Earthquake Data Center
- `stead`: **Recommended** - Comprehensive STanford EArthquake Dataset
- `geofon`: GEOFON network data
- `neic`: National Earthquake Information Center

**EQTransformer Versions:**
- `original`, `ethz`, `instance`, `scedc`, `stead`, `geofon`

**GPD Versions:**
- `original`, `ethz`, `scedc`, `stead`, `geofon`, `neic`

### Picker Parameters
- **threshold** (0-1): Lower = more sensitive, more false positives
- **min_duration** (samples): Minimum pick duration to keep
- **merge_distance** (samples): Merge picks within this distance

### Classifier Parameters
- **window_duration**: Must be 100s for QuakeXNet (5000 samples @ 50Hz)
- **stride**: Smaller = more windows, better coverage, slower
- **batch_size**: Larger = faster, needs more memory (12 is good default)
- **include_sliding**: True = classify entire stream with sliding windows
- **include_event_centered**: True = classify windows centered on picks

### Merging Parameters
- **time_tolerance**: Matching window in seconds
  - ±5s: Tight coupling, good for earthquakes
  - ±10s: Moderate, recommended default
  - ±20s: Loose, better for emergent surface events

## Common Workflows

### Landslide Detection
```python
# Focus on surface events
surface_events = df_merged[df_merged['class_label'] == 'su']
surface_events_clean = surface_events[surface_events['class_prob'] > 0.6]
```

### Earthquake Catalog
```python
# Get earthquakes with picks
eq_with_picks = df_merged[
    (df_merged['class_label'] == 'eq') & 
    (df_merged['match_type'] == 'matched')
]
```

### Events Missed by Picker
```python
# Classifications without picks (picker missed these)
missed_events = df_merged[df_merged['match_type'] == 'class_only']
non_noise_missed = missed_events[missed_events['class_label'] != 'no']
```

## Troubleshooting

### Weight Download Fails
```python
# Manually specify weights path
classifier = create_classifier(
    weights_path='/path/to/quakexnet_base.pt',
    auto_download=False
)
```

### Memory Issues
```python
# Reduce batch size
classifications = classify_waveform_windows(
    stream, classifier,
    batch_size=4,  # Smaller batches
    include_sliding=True,
    include_event_centered=False  # Skip event-centered
)
```

### Too Many/Few Detections
```python
# Adjust thresholds
pick_detections = multi_class_detection(
    pick_probabilities,
    threshold=0.7,  # Higher = fewer, more confident picks
)

# Filter by probability
high_conf = df_merged[df_merged['class_prob'] > 0.8]
```

## Export Results

```python
# Save merged results
df_merged.to_csv('merged_events.csv', index=False)

# Save filtered results
surface_events.to_csv('surface_events.csv', index=False)
earthquakes.to_csv('earthquakes.csv', index=False)

# Export for further analysis
import json
event_dict = df_merged.to_dict('records')
with open('events.json', 'w') as f:
    json.dump(event_dict, f, indent=2)
```
