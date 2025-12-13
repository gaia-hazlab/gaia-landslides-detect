"""
Event detection utilities using SeisBench workflow.

This module implements seismic event detection following the SeisBench approach,
with support for pre-trained models like PhaseNet, EQTransformer, and GPD.
"""

import numpy as np
import torch
from scipy import signal
from typing import Dict, List, Tuple, Optional


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
