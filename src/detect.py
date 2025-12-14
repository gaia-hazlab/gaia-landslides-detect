"""
Event detection utilities using SeisBench workflow.

This module implements seismic event detection following the SeisBench approach,
with support for pre-trained models like PhaseNet, EQTransformer, and GPD.
"""

import numpy as np
import torch
from scipy import signal
from typing import Dict, List, Tuple, Optional
from obspy import Stream, UTCDateTime
import pandas as pd
from tqdm import tqdm


def smooth_moving_avg(data: np.ndarray, window_size: int = 100) -> np.ndarray:
    """
    Apply moving average smoothing to probability predictions.
    
    Args:
        data (np.ndarray): Input probability array
        window_size (int): Window size for moving average
        
    Returns:
        np.ndarray: Smoothed probability array
    """
    if len(data) < window_size:
        return data
    
    # Use convolve for efficient moving average
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(data, kernel, mode='same')
    
    return smoothed


def detect_event_windows(
    probabilities: np.ndarray,
    threshold: float = 0.5,
    min_duration: int = 10,
    merge_distance: int = 50
) -> List[Dict]:
    """
    Detect event windows from probability predictions.
    
    Args:
        probabilities (np.ndarray): Probability array (1D)
        threshold (float): Detection threshold (0-1)
        min_duration (int): Minimum event duration in samples
        merge_distance (int): Maximum gap between events to merge
        
    Returns:
        List[Dict]: List of detected events with metadata
    """
    # Find samples above threshold
    above_threshold = probabilities > threshold
    
    # Find start and end indices of continuous segments
    diff = np.diff(above_threshold.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    
    # Handle edge cases
    if above_threshold[0]:
        starts = np.concatenate([[0], starts])
    if above_threshold[-1]:
        ends = np.concatenate([ends, [len(probabilities)]])
    
    # Filter by minimum duration
    durations = ends - starts
    valid = durations >= min_duration
    starts = starts[valid]
    ends = ends[valid]
    
    # Merge nearby events
    if len(starts) > 1:
        merged_starts = [starts[0]]
        merged_ends = [ends[0]]
        
        for i in range(1, len(starts)):
            # Check gap between end of previous event and start of current event
            if starts[i] - merged_ends[-1] <= merge_distance:
                # Merge with previous event by extending its end time
                merged_ends[-1] = max(merged_ends[-1], ends[i])
            else:
                # Start new event
                merged_starts.append(starts[i])
                merged_ends.append(ends[i])
        
        starts = np.array(merged_starts)
        ends = np.array(merged_ends)
    
    # Calculate event metrics
    events = []
    for start, end in zip(starts, ends):
        event_probs = probabilities[start:end]
        
        events.append({
            'start': int(start),
            'end': int(end),
            'duration': int(end - start),
            'max_prob': float(np.max(event_probs)),
            'mean_prob': float(np.mean(event_probs)),
            'area_under_curve': float(np.trapz(event_probs)),
        })
    
    return events


def calculate_event_metrics(
    probabilities: np.ndarray,
    events: List[Dict],
    sampling_rate: float = 100.0
) -> List[Dict]:
    """
    Calculate additional metrics for detected events.
    
    Args:
        probabilities (np.ndarray): Original probability array
        events (List[Dict]): List of detected events
        sampling_rate (float): Sampling rate in Hz
        
    Returns:
        List[Dict]: Events with additional metrics
    """
    enhanced_events = []
    
    for event in events:
        start = event['start']
        end = event['end']
        event_probs = probabilities[start:end]
        
        # Add temporal information
        enhanced_event = event.copy()
        enhanced_event['start_time'] = start / sampling_rate
        enhanced_event['end_time'] = end / sampling_rate
        enhanced_event['duration_sec'] = (end - start) / sampling_rate
        
        # Add peak information
        peak_idx = np.argmax(event_probs)
        enhanced_event['peak_time'] = (start + peak_idx) / sampling_rate
        enhanced_event['peak_idx'] = int(start + peak_idx)
        
        # Add signal quality metrics
        enhanced_event['std_prob'] = float(np.std(event_probs))
        enhanced_event['median_prob'] = float(np.median(event_probs))
        
        enhanced_events.append(enhanced_event)
    
    return enhanced_events


def apply_nms(
    events: List[Dict],
    iou_threshold: float = 0.3
) -> List[Dict]:
    """
    Apply Non-Maximum Suppression to remove overlapping detections.
    
    Args:
        events (List[Dict]): List of detected events
        iou_threshold (float): IoU threshold for suppression
        
    Returns:
        List[Dict]: Filtered events after NMS
    """
    if len(events) <= 1:
        return events
    
    # Sort by max_prob in descending order
    sorted_events = sorted(events, key=lambda x: x['max_prob'], reverse=True)
    
    keep = []
    while sorted_events:
        # Take the event with highest probability
        current = sorted_events.pop(0)
        keep.append(current)
        
        # Remove overlapping events
        filtered = []
        for event in sorted_events:
            iou = calculate_iou(current, event)
            if iou < iou_threshold:
                filtered.append(event)
        
        sorted_events = filtered
    
    # Sort by start time
    keep = sorted(keep, key=lambda x: x['start'])
    
    return keep


def calculate_iou(event1: Dict, event2: Dict) -> float:
    """
    Calculate Intersection over Union between two events.
    
    Args:
        event1 (Dict): First event
        event2 (Dict): Second event
        
    Returns:
        float: IoU value (0-1)
    """
    start1, end1 = event1['start'], event1['end']
    start2, end2 = event2['start'], event2['end']
    
    # Calculate intersection
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    intersection = max(0, intersection_end - intersection_start)
    
    # Calculate union
    union = (end1 - start1) + (end2 - start2) - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def multi_class_detection(
    probabilities: Dict[str, np.ndarray],
    threshold: float = 0.5,
    min_duration: int = 10,
    merge_distance: int = 50,
    apply_smoothing: bool = True,
    smooth_window: int = 100
) -> Dict[str, List[Dict]]:
    """
    Perform multi-class event detection (e.g., earthquake, explosion, surface event).
    
    Args:
        probabilities (Dict[str, np.ndarray]): Dictionary of class probabilities
        threshold (float): Detection threshold
        min_duration (int): Minimum event duration in samples
        merge_distance (int): Maximum gap to merge events
        apply_smoothing (bool): Whether to apply smoothing
        smooth_window (int): Smoothing window size
        
    Returns:
        Dict[str, List[Dict]]: Detected events per class
    """
    detections = {}
    
    for class_name, probs in probabilities.items():
        # Apply smoothing if requested
        if apply_smoothing:
            probs_smoothed = smooth_moving_avg(probs, window_size=smooth_window)
        else:
            probs_smoothed = probs
        
        # Detect events
        events = detect_event_windows(
            probs_smoothed,
            threshold=threshold,
            min_duration=min_duration,
            merge_distance=merge_distance
        )
        
        detections[class_name] = events
    
    return detections


def extract_fixed_windows(
    stream: Stream,
    window_duration: float = 100.0,
    stride: float = 50.0,
    target_sampling_rate: float = 50.0
) -> Tuple[List[np.ndarray], List[float], List[int], List[int]]:
    """
    Extract fixed-duration sliding windows from a stream.
    
    Args:
        stream (Stream): ObsPy stream containing waveform data
        window_duration (float): Window duration in seconds (default: 100s for QuakeXNet)
        stride (float): Stride between windows in seconds (default: 50s)
        target_sampling_rate (float): Target sampling rate in Hz (default: 50Hz)
        
    Returns:
        Tuple containing:
            - windows (List[np.ndarray]): List of windowed data arrays (n_components, n_samples)
            - center_times (List[float]): Center time of each window (seconds from stream start)
            - start_indices (List[int]): Start sample index of each window
            - end_indices (List[int]): End sample index of each window
    """
    # Get stream parameters
    st = stream.copy()
    st.detrend('linear')
    st.merge(fill_value=0)
    
    # Resample if needed
    current_sr = st[0].stats.sampling_rate
    if abs(current_sr - target_sampling_rate) > 0.1:
        st.resample(target_sampling_rate)
    
    sampling_rate = st[0].stats.sampling_rate
    window_samples = int(window_duration * sampling_rate)
    stride_samples = int(stride * sampling_rate)
    
    # Get total number of samples
    n_samples = min([len(tr.data) for tr in st])
    
    # Calculate number of windows
    n_windows = max(1, (n_samples - window_samples) // stride_samples + 1)
    
    windows = []
    center_times = []
    start_indices = []
    end_indices = []
    
    for i in range(n_windows):
        start_idx = i * stride_samples
        end_idx = start_idx + window_samples
        
        # Handle boundary case with padding
        if end_idx > n_samples:
            # Pad with zeros
            window_data = []
            for tr in st:
                data = tr.data[start_idx:n_samples]
                padding = np.zeros(end_idx - n_samples)
                padded_data = np.concatenate([data, padding])
                window_data.append(padded_data)
        else:
            window_data = [tr.data[start_idx:end_idx] for tr in st]
        
        window_array = np.array(window_data)
        windows.append(window_array)
        
        # Calculate center time
        center_idx = start_idx + window_samples // 2
        center_time = center_idx / sampling_rate
        center_times.append(center_time)
        
        start_indices.append(start_idx)
        end_indices.append(min(end_idx, n_samples))
    
    return windows, center_times, start_indices, end_indices


def extract_event_centered_windows(
    stream: Stream,
    event_times: List[float],
    window_duration: float = 100.0,
    target_sampling_rate: float = 50.0
) -> Tuple[List[np.ndarray], List[float], List[int], List[int]]:
    """
    Extract fixed-duration windows centered on event times.
    
    Args:
        stream (Stream): ObsPy stream containing waveform data
        event_times (List[float]): List of event times in seconds from stream start
        window_duration (float): Window duration in seconds (default: 100s)
        target_sampling_rate (float): Target sampling rate in Hz (default: 50Hz)
        
    Returns:
        Tuple containing:
            - windows (List[np.ndarray]): List of windowed data arrays
            - center_times (List[float]): Center time of each window
            - start_indices (List[int]): Start sample index of each window
            - end_indices (List[int]): End sample index of each window
    """
    # Get stream parameters
    st = stream.copy()
    st.detrend('linear')
    st.merge(fill_value=0)
    
    # Resample if needed
    current_sr = st[0].stats.sampling_rate
    if abs(current_sr - target_sampling_rate) > 0.1:
        st.resample(target_sampling_rate)
    
    sampling_rate = st[0].stats.sampling_rate
    window_samples = int(window_duration * sampling_rate)
    half_window = window_samples // 2
    
    n_samples = min([len(tr.data) for tr in st])
    
    windows = []
    center_times = []
    start_indices = []
    end_indices = []
    
    for event_time in event_times:
        center_idx = int(event_time * sampling_rate)
        start_idx = center_idx - half_window
        end_idx = center_idx + half_window
        
        # Handle boundary cases
        if start_idx < 0:
            # Pad at beginning
            window_data = []
            for tr in st:
                data = tr.data[0:end_idx]
                padding = np.zeros(-start_idx)
                padded_data = np.concatenate([padding, data])
                window_data.append(padded_data)
            actual_start = 0
        elif end_idx > n_samples:
            # Pad at end
            window_data = []
            for tr in st:
                data = tr.data[start_idx:n_samples]
                padding = np.zeros(end_idx - n_samples)
                padded_data = np.concatenate([data, padding])
                window_data.append(padded_data)
            actual_start = start_idx
        else:
            # Normal case
            window_data = [tr.data[start_idx:end_idx] for tr in st]
            actual_start = start_idx
        
        window_array = np.array(window_data)
        windows.append(window_array)
        center_times.append(event_time)
        start_indices.append(max(0, start_idx))
        end_indices.append(min(end_idx, n_samples))
    
    return windows, center_times, start_indices, end_indices


def classify_waveform_windows(
    stream: Stream,
    classifier,
    picker_detections: Optional[Dict[str, List[Dict]]] = None,
    window_duration: float = 100.0,
    stride: float = 50.0,
    batch_size: int = 12,
    include_sliding: bool = True,
    include_event_centered: bool = True
) -> List[Dict]:
    """
    Classify waveform windows using hybrid approach (sliding + event-centered).
    
    Args:
        stream (Stream): ObsPy stream containing waveform data
        classifier: SeismicClassifier instance with classify_windows method
        picker_detections (Dict): Optional picker detections with event times
        window_duration (float): Window duration in seconds (default: 100s)
        stride (float): Stride for sliding windows in seconds (default: 50s)
        batch_size (int): Batch size for processing (default: 12)
        include_sliding (bool): Include sliding window classification
        include_event_centered (bool): Include event-centered classification
        
    Returns:
        List[Dict]: Classification results with keys: window_start, window_end,
                    center_time, class_label, class_prob_eq, class_prob_px,
                    class_prob_no, class_prob_su, window_type
    """
    all_results = []
    
    # 1. Sliding window classification
    if include_sliding:
        windows, center_times, start_indices, end_indices = extract_fixed_windows(
            stream, window_duration=window_duration, stride=stride
        )
        
        if windows:
            print(f"Processing {len(windows)} sliding windows...")
            classifications = classifier.classify_windows(windows, batch_size=batch_size)
            
            for i, (class_probs, class_label) in enumerate(classifications):
                result = {
                    'window_start': start_indices[i],
                    'window_end': end_indices[i],
                    'center_time': center_times[i],
                    'class_label': class_label,
                    'class_prob_eq': float(class_probs[0]),
                    'class_prob_px': float(class_probs[1]),
                    'class_prob_no': float(class_probs[2]),
                    'class_prob_su': float(class_probs[3]),
                    'window_type': 'sliding'
                }
                all_results.append(result)
    
    # 2. Event-centered classification
    if include_event_centered and picker_detections:
        # Extract event times from picker detections
        event_times = []
        for class_name, events in picker_detections.items():
            for event in events:
                if 'peak_idx' in event:
                    # Use peak time if available
                    sampling_rate = stream[0].stats.sampling_rate
                    event_time = event['peak_idx'] / sampling_rate
                    event_times.append(event_time)
                elif 'peak_time' in event:
                    event_times.append(event['peak_time'])
        
        if event_times:
            # Remove duplicates and sort
            event_times = sorted(list(set(event_times)))
            
            windows, center_times, start_indices, end_indices = extract_event_centered_windows(
                stream, event_times, window_duration=window_duration
            )
            
            print(f"Processing {len(windows)} event-centered windows...")
            classifications = classifier.classify_windows(windows, batch_size=batch_size)
            
            for i, (class_probs, class_label) in enumerate(classifications):
                result = {
                    'window_start': start_indices[i],
                    'window_end': end_indices[i],
                    'center_time': center_times[i],
                    'class_label': class_label,
                    'class_prob_eq': float(class_probs[0]),
                    'class_prob_px': float(class_probs[1]),
                    'class_prob_no': float(class_probs[2]),
                    'class_prob_su': float(class_probs[3]),
                    'window_type': 'event_centered'
                }
                all_results.append(result)
    
    return all_results


def merge_picks_and_classifications(
    picker_results: List[Dict],
    classification_results: List[Dict],
    time_tolerance: float = 10.0,
    sampling_rate: float = 100.0
) -> pd.DataFrame:
    """
    Merge picker results with classification results based on temporal overlap.
    
    When multiple classifications overlap with a pick, takes the closest temporal match.
    Preserves all unmatched picks and classifications.
    
    Args:
        picker_results (List[Dict]): Picker detections with pick_time, pick_phase, etc.
        classification_results (List[Dict]): Classification windows with center_time, class_label, etc.
        time_tolerance (float): Temporal tolerance for matching in seconds (default: ±10s)
        sampling_rate (float): Sampling rate for converting indices to times
        
    Returns:
        pd.DataFrame: Merged results with columns: pick_time, pick_phase, pick_prob,
                      class_label, class_prob, match_type, time_diff
    """
    merged_records = []
    matched_classifications = set()
    
    # Process each pick
    for pick in picker_results:
        # Get pick time (handle different formats)
        if 'pick_time' in pick:
            pick_time = pick['pick_time']
        elif 'peak_time' in pick:
            pick_time = pick['peak_time']
        elif 'peak_idx' in pick:
            pick_time = pick['peak_idx'] / sampling_rate
        else:
            continue
        
        pick_phase = pick.get('phase', pick.get('class_name', 'unknown'))
        pick_prob = pick.get('max_prob', pick.get('pick_prob', 0.0))
        
        # Find matching classifications within tolerance
        matches = []
        for i, classification in enumerate(classification_results):
            class_time = classification['center_time']
            time_diff = abs(class_time - pick_time)
            
            if time_diff <= time_tolerance:
                matches.append((i, time_diff, classification))
        
        if matches:
            # Take closest match
            matches.sort(key=lambda x: x[1])
            idx, time_diff, best_match = matches[0]
            matched_classifications.add(idx)
            
            # Create merged record
            record = {
                'pick_time': pick_time,
                'pick_phase': pick_phase,
                'pick_prob': pick_prob,
                'class_label': best_match['class_label'],
                'class_prob': max([
                    best_match['class_prob_eq'],
                    best_match['class_prob_px'],
                    best_match['class_prob_no'],
                    best_match['class_prob_su']
                ]),
                'class_prob_eq': best_match['class_prob_eq'],
                'class_prob_px': best_match['class_prob_px'],
                'class_prob_no': best_match['class_prob_no'],
                'class_prob_su': best_match['class_prob_su'],
                'match_type': 'matched',
                'time_diff': time_diff,
                'window_type': best_match.get('window_type', 'unknown')
            }
        else:
            # Unmatched pick
            record = {
                'pick_time': pick_time,
                'pick_phase': pick_phase,
                'pick_prob': pick_prob,
                'class_label': None,
                'class_prob': None,
                'class_prob_eq': None,
                'class_prob_px': None,
                'class_prob_no': None,
                'class_prob_su': None,
                'match_type': 'pick_only',
                'time_diff': None,
                'window_type': None
            }
        
        merged_records.append(record)
    
    # Add unmatched classifications
    for i, classification in enumerate(classification_results):
        if i not in matched_classifications:
            record = {
                'pick_time': None,
                'pick_phase': None,
                'pick_prob': None,
                'class_label': classification['class_label'],
                'class_prob': max([
                    classification['class_prob_eq'],
                    classification['class_prob_px'],
                    classification['class_prob_no'],
                    classification['class_prob_su']
                ]),
                'class_prob_eq': classification['class_prob_eq'],
                'class_prob_px': classification['class_prob_px'],
                'class_prob_no': classification['class_prob_no'],
                'class_prob_su': classification['class_prob_su'],
                'match_type': 'class_only',
                'time_diff': None,
                'window_type': classification.get('window_type', 'unknown')
            }
            merged_records.append(record)
    
    # Create DataFrame and sort by time
    df = pd.DataFrame(merged_records)
    
    # Sort by pick_time (with NaN last) or class center time
    if not df.empty:
        df['sort_time'] = df['pick_time'].fillna(df.index.map(
            lambda i: classification_results[i]['center_time'] 
            if i < len(classification_results) else 0
        ))
        df = df.sort_values('sort_time').drop('sort_time', axis=1)
        df = df.reset_index(drop=True)
    
    return df


def filter_noise_events(df: pd.DataFrame, remove_noise: bool = True) -> pd.DataFrame:
    """
    Convenience function to filter noise from merged event DataFrame.
    
    Args:
        df (pd.DataFrame): Merged events DataFrame
        remove_noise (bool): If True, remove events classified as noise
        
    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    if remove_noise:
        return df[df['class_label'] != 'no'].copy()
    return df.copy()


def get_event_type_counts(df: pd.DataFrame) -> Dict[str, int]:
    """
    Get counts of events by type from merged DataFrame.
    
    Args:
        df (pd.DataFrame): Merged events DataFrame
        
    Returns:
        Dict[str, int]: Dictionary with counts by event type
    """
    counts = {
        'total': len(df),
        'matched': len(df[df['match_type'] == 'matched']),
        'pick_only': len(df[df['match_type'] == 'pick_only']),
        'class_only': len(df[df['match_type'] == 'class_only']),
    }
    
    # Add class-specific counts
    for class_label in ['eq', 'px', 'no', 'su']:
        counts[class_label] = len(df[df['class_label'] == class_label])
    
    # Add phase-specific counts
    for phase in df['pick_phase'].dropna().unique():
        counts[f'phase_{phase}'] = len(df[df['pick_phase'] == phase])
    
    return counts



