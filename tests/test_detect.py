"""
Unit tests for event detection utilities.

To run these tests:
    pytest tests/test_detect.py
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detect import (
    smooth_moving_avg,
    detect_event_windows,
    calculate_event_metrics,
    calculate_iou,
    apply_nms,
    multi_class_detection
)


class TestSmoothMovingAvg:
    """Tests for moving average smoothing."""
    
    def test_smooth_basic(self):
        """Test basic smoothing operation."""
        data = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])
        smoothed = smooth_moving_avg(data, window_size=3)
        
        assert len(smoothed) == len(data)
        assert isinstance(smoothed, np.ndarray)
    
    def test_smooth_short_data(self):
        """Test smoothing with data shorter than window."""
        data = np.array([1, 2, 3])
        smoothed = smooth_moving_avg(data, window_size=10)
        
        assert len(smoothed) == len(data)
        np.testing.assert_array_equal(smoothed, data)
    
    def test_smooth_reduces_noise(self):
        """Test that smoothing reduces high-frequency noise."""
        np.random.seed(42)
        signal = np.sin(np.linspace(0, 4*np.pi, 100))
        noise = np.random.randn(100) * 0.1
        noisy_signal = signal + noise
        
        smoothed = smooth_moving_avg(noisy_signal, window_size=10)
        
        # Smoothed should be closer to original signal
        original_error = np.mean((noisy_signal - signal) ** 2)
        smoothed_error = np.mean((smoothed - signal) ** 2)
        assert smoothed_error < original_error


class TestDetectEventWindows:
    """Tests for event window detection."""
    
    def test_detect_simple_event(self):
        """Test detection of simple event above threshold."""
        probs = np.zeros(100)
        probs[30:50] = 0.8  # Event from index 30 to 50
        
        events = detect_event_windows(probs, threshold=0.5, min_duration=10)
        
        assert len(events) == 1
        assert events[0]['start'] == 30
        assert events[0]['end'] == 50
        assert events[0]['duration'] == 20
    
    def test_detect_multiple_events(self):
        """Test detection of multiple events."""
        probs = np.zeros(200)
        probs[20:40] = 0.8   # Event 1
        probs[100:130] = 0.9  # Event 2
        
        events = detect_event_windows(probs, threshold=0.5, min_duration=10)
        
        assert len(events) == 2
        assert events[0]['start'] == 20
        assert events[1]['start'] == 100
    
    def test_min_duration_filter(self):
        """Test that short events are filtered out."""
        probs = np.zeros(100)
        probs[30:35] = 0.8   # 5 samples (too short)
        probs[50:65] = 0.8   # 15 samples (long enough)
        
        events = detect_event_windows(probs, threshold=0.5, min_duration=10)
        
        assert len(events) == 1
        assert events[0]['start'] == 50
    
    def test_merge_nearby_events(self):
        """Test merging of nearby events."""
        probs = np.zeros(200)
        probs[20:40] = 0.8
        probs[50:70] = 0.8  # Only 10 samples gap
        
        # Without merging (large merge_distance)
        events_merged = detect_event_windows(probs, threshold=0.5, merge_distance=100)
        assert len(events_merged) == 1  # Should merge
        
        # With small merge distance
        events_separate = detect_event_windows(probs, threshold=0.5, merge_distance=5)
        assert len(events_separate) == 2  # Should not merge
    
    def test_event_metrics(self):
        """Test that event metrics are calculated correctly."""
        probs = np.zeros(100)
        probs[30:50] = 0.8
        
        events = detect_event_windows(probs, threshold=0.5)
        
        assert len(events) == 1
        event = events[0]
        
        assert 'max_prob' in event
        assert 'mean_prob' in event
        assert 'area_under_curve' in event
        assert event['max_prob'] == pytest.approx(0.8)
        assert event['mean_prob'] == pytest.approx(0.8)


class TestCalculateEventMetrics:
    """Tests for event metric calculation."""
    
    def test_calculate_metrics(self):
        """Test calculation of enhanced event metrics."""
        probs = np.random.rand(1000)
        events = [
            {'start': 100, 'end': 200, 'max_prob': 0.9, 'mean_prob': 0.7}
        ]
        
        enhanced = calculate_event_metrics(probs, events, sampling_rate=100.0)
        
        assert len(enhanced) == 1
        event = enhanced[0]
        
        assert 'start_time' in event
        assert 'end_time' in event
        assert 'duration_sec' in event
        assert 'peak_time' in event
        assert 'peak_idx' in event
        assert event['start_time'] == pytest.approx(1.0)
        assert event['end_time'] == pytest.approx(2.0)
        assert event['duration_sec'] == pytest.approx(1.0)


class TestCalculateIoU:
    """Tests for IoU calculation."""
    
    def test_iou_no_overlap(self):
        """Test IoU with no overlap."""
        event1 = {'start': 0, 'end': 10}
        event2 = {'start': 20, 'end': 30}
        
        iou = calculate_iou(event1, event2)
        assert iou == 0.0
    
    def test_iou_complete_overlap(self):
        """Test IoU with complete overlap."""
        event1 = {'start': 0, 'end': 10}
        event2 = {'start': 0, 'end': 10}
        
        iou = calculate_iou(event1, event2)
        assert iou == 1.0
    
    def test_iou_partial_overlap(self):
        """Test IoU with partial overlap."""
        event1 = {'start': 0, 'end': 10}
        event2 = {'start': 5, 'end': 15}
        
        iou = calculate_iou(event1, event2)
        # Intersection: 5, Union: 15
        assert iou == pytest.approx(5.0 / 15.0)


class TestApplyNMS:
    """Tests for Non-Maximum Suppression."""
    
    def test_nms_removes_overlapping(self):
        """Test that NMS removes overlapping low-confidence events."""
        events = [
            {'start': 0, 'end': 10, 'max_prob': 0.9},
            {'start': 5, 'end': 15, 'max_prob': 0.7},  # Overlaps with first
        ]
        
        filtered = apply_nms(events, iou_threshold=0.3)
        
        assert len(filtered) == 1
        assert filtered[0]['max_prob'] == 0.9
    
    def test_nms_keeps_non_overlapping(self):
        """Test that NMS keeps non-overlapping events."""
        events = [
            {'start': 0, 'end': 10, 'max_prob': 0.9},
            {'start': 20, 'end': 30, 'max_prob': 0.7},  # No overlap
        ]
        
        filtered = apply_nms(events, iou_threshold=0.3)
        
        assert len(filtered) == 2


class TestMultiClassDetection:
    """Tests for multi-class detection."""
    
    def test_multi_class_detection(self):
        """Test detection across multiple classes."""
        probabilities = {
            'eq': np.random.rand(1000),
            'px': np.random.rand(1000),
        }
        
        # Add some clear events
        probabilities['eq'][100:150] = 0.9
        probabilities['px'][200:250] = 0.8
        
        detections = multi_class_detection(
            probabilities,
            threshold=0.5,
            min_duration=10
        )
        
        assert 'eq' in detections
        assert 'px' in detections
        assert len(detections['eq']) >= 1
        assert len(detections['px']) >= 1
    
    def test_multi_class_with_smoothing(self):
        """Test multi-class detection with smoothing enabled."""
        probabilities = {
            'eq': np.random.rand(1000) * 0.3,  # Low baseline
        }
        probabilities['eq'][100:150] = 0.9
        
        detections = multi_class_detection(
            probabilities,
            threshold=0.5,
            apply_smoothing=True,
            smooth_window=10
        )
        
        assert 'eq' in detections
        assert len(detections['eq']) >= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
