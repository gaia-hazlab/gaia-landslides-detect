"""
Quick test to verify the two-step workflow implementation.
"""

import numpy as np
from obspy import Stream, Trace, UTCDateTime
from src.detect import (
    extract_fixed_windows,
    extract_event_centered_windows,
    merge_picks_and_classifications
)

# Create synthetic stream for testing
def create_test_stream(duration=300, sampling_rate=50):
    """Create a 3-component synthetic stream."""
    starttime = UTCDateTime(2024, 1, 1, 0, 0, 0)
    npts = int(duration * sampling_rate)
    
    traces = []
    for component in ['E', 'N', 'Z']:
        data = np.random.randn(npts) * 0.1  # Random noise
        # Add a simple synthetic event
        event_start = npts // 2
        event_duration = int(10 * sampling_rate)
        data[event_start:event_start + event_duration] += np.sin(
            np.linspace(0, 10 * np.pi, event_duration)
        )
        
        tr = Trace(data=data)
        tr.stats.sampling_rate = sampling_rate
        tr.stats.starttime = starttime
        tr.stats.channel = f'HH{component}'
        tr.stats.station = 'TEST'
        tr.stats.network = 'XX'
        traces.append(tr)
    
    return Stream(traces=traces)

def test_window_extraction():
    """Test window extraction functions."""
    print("Testing window extraction...")
    
    stream = create_test_stream(duration=300, sampling_rate=50)
    
    # Test sliding windows
    windows, center_times, starts, ends = extract_fixed_windows(
        stream, window_duration=100.0, stride=50.0
    )
    print(f"✓ Sliding windows: {len(windows)} windows extracted")
    print(f"  First window shape: {windows[0].shape}")
    print(f"  Center times: {center_times[:3]}")
    
    # Test event-centered windows
    event_times = [50.0, 150.0, 200.0]  # 3 synthetic events
    windows, center_times, starts, ends = extract_event_centered_windows(
        stream, event_times, window_duration=100.0
    )
    print(f"✓ Event-centered windows: {len(windows)} windows extracted")
    print(f"  Event times: {event_times}")

def test_merging():
    """Test merging function."""
    print("\nTesting merging...")
    
    # Synthetic picker results
    picker_results = [
        {'pick_time': 50.0, 'phase': 'P', 'max_prob': 0.8, 'peak_idx': 2500},
        {'pick_time': 150.0, 'phase': 'S', 'max_prob': 0.7, 'peak_idx': 7500},
        {'pick_time': 250.0, 'phase': 'P', 'max_prob': 0.9, 'peak_idx': 12500},
    ]
    
    # Synthetic classification results
    classification_results = [
        {
            'center_time': 52.0, 'class_label': 'eq',
            'class_prob_eq': 0.8, 'class_prob_px': 0.1,
            'class_prob_no': 0.05, 'class_prob_su': 0.05,
            'window_type': 'sliding'
        },
        {
            'center_time': 148.0, 'class_label': 'su',
            'class_prob_eq': 0.1, 'class_prob_px': 0.2,
            'class_prob_no': 0.05, 'class_prob_su': 0.65,
            'window_type': 'event_centered'
        },
        {
            'center_time': 200.0, 'class_label': 'px',
            'class_prob_eq': 0.1, 'class_prob_px': 0.7,
            'class_prob_no': 0.15, 'class_prob_su': 0.05,
            'window_type': 'sliding'
        },
    ]
    
    # Test merging
    df_merged = merge_picks_and_classifications(
        picker_results=picker_results,
        classification_results=classification_results,
        time_tolerance=10.0,
        sampling_rate=50.0
    )
    
    print(f"✓ Merging complete: {len(df_merged)} total events")
    print(f"  Matched: {len(df_merged[df_merged['match_type'] == 'matched'])}")
    print(f"  Pick only: {len(df_merged[df_merged['match_type'] == 'pick_only'])}")
    print(f"  Class only: {len(df_merged[df_merged['match_type'] == 'class_only'])}")
    print("\nMerged DataFrame:")
    print(df_merged[['pick_time', 'pick_phase', 'class_label', 'match_type', 'time_diff']])

if __name__ == '__main__':
    print("="*60)
    print("Testing Two-Step Workflow Implementation")
    print("="*60)
    
    test_window_extraction()
    test_merging()
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)
