"""
Interactive plotting utilities for seismic event detection quality control.

This module provides interactive visualization tools for reviewing detection results,
including waveforms, probability predictions, and event windows.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
from scipy import signal
from obspy import Stream, UTCDateTime
from typing import Dict, List, Optional, Tuple
import warnings


def plot_detection_results(
    stream: Stream,
    probabilities: Dict[str, np.ndarray],
    events: Optional[Dict[str, List[Dict]]] = None,
    sampling_rate: float = 100.0,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None
):
    """
    Plot seismic waveforms with detection probabilities and event windows.
    
    Args:
        stream (Stream): Original seismic stream
        probabilities (Dict[str, np.ndarray]): Probability predictions per class
        events (Dict[str, List[Dict]]): Detected events per class
        sampling_rate (float): Sampling rate in Hz
        figsize (tuple): Figure size
        save_path (str): Optional path to save figure
    """
    n_traces = len(stream)
    n_classes = len(probabilities)
    n_rows = n_traces + n_classes + 1  # waveforms + probabilities + spectrogram
    
    fig, axes = plt.subplots(n_rows, 1, figsize=figsize, sharex=True)
    if n_rows == 1:
        axes = [axes]
    
    # Time array
    n_samples = len(stream[0].data)
    time = np.arange(n_samples) / sampling_rate
    
    # Plot waveforms
    for i, trace in enumerate(stream):
        ax = axes[i]
        ax.plot(time, trace.data, 'k', linewidth=0.5, label=trace.stats.channel)
        ax.set_ylabel('Amplitude')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    # Plot probabilities
    colors = {'eq': 'red', 'px': 'blue', 'su': 'green', 'noise': 'gray'}
    for idx, (class_name, probs) in enumerate(probabilities.items()):
        ax = axes[n_traces + idx]
        
        # Adjust time array if probability array is different size
        prob_time = np.linspace(0, time[-1], len(probs))
        
        color = colors.get(class_name, 'black')
        ax.plot(prob_time, probs, color=color, linewidth=1.5, label=class_name.upper())
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_ylabel('Probability')
        ax.set_ylim([0, 1])
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Plot event windows if provided
        if events and class_name in events:
            for event in events[class_name]:
                start_time = event['start'] / sampling_rate if 'start' in event else event.get('start_time', 0)
                end_time = event['end'] / sampling_rate if 'end' in event else event.get('end_time', 0)
                ax.axvspan(start_time, end_time, alpha=0.2, color=color)
    
    # Add spectrogram for the first trace
    if n_traces > 0:
        ax = axes[-1]
        trace = stream[0]
        f, t, Sxx = signal.spectrogram(trace.data, fs=sampling_rate, nperseg=256)
        ax.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='viridis')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_ylim([0, min(50, sampling_rate/2)])
    
    axes[-1].set_xlabel('Time (s)')
    
    # Add title with station info
    if stream:
        stats = stream[0].stats
        title = f"Station: {stats.network}.{stats.station} | "
        title += f"Start: {stats.starttime.strftime('%Y-%m-%d %H:%M:%S')}"
        fig.suptitle(title, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig, axes


def interactive_detection_viewer(
    stream: Stream,
    probabilities: Dict[str, np.ndarray],
    events: Dict[str, List[Dict]],
    sampling_rate: float = 100.0
):
    """
    Create an interactive viewer for detection results.
    
    Args:
        stream (Stream): Original seismic stream
        probabilities (Dict[str, np.ndarray]): Probability predictions per class
        events (Dict[str, List[Dict]]): Detected events per class
        sampling_rate (float): Sampling rate in Hz
    """
    from scipy import signal as scipy_signal
    
    fig = plt.figure(figsize=(16, 10))
    
    # Create subplots
    gs = fig.add_gridspec(4, 1, hspace=0.3, height_ratios=[2, 1, 1, 1])
    ax_wave = fig.add_subplot(gs[0])
    ax_prob = fig.add_subplot(gs[1], sharex=ax_wave)
    ax_spec = fig.add_subplot(gs[2], sharex=ax_wave)
    ax_events = fig.add_subplot(gs[3])
    
    # Time arrays
    n_samples = len(stream[0].data)
    time = np.arange(n_samples) / sampling_rate
    
    # Plot waveform
    trace = stream[0]
    line_wave, = ax_wave.plot(time, trace.data, 'k', linewidth=0.5)
    ax_wave.set_ylabel('Amplitude')
    ax_wave.set_title(f'{trace.stats.network}.{trace.stats.station}.{trace.stats.channel}')
    ax_wave.grid(True, alpha=0.3)
    
    # Plot probabilities
    colors = {'eq': 'red', 'px': 'blue', 'su': 'green'}
    prob_lines = {}
    for class_name, probs in probabilities.items():
        prob_time = np.linspace(0, time[-1], len(probs))
        color = colors.get(class_name, 'black')
        line, = ax_prob.plot(prob_time, probs, color=color, linewidth=1.5, 
                            label=class_name.upper(), alpha=0.7)
        prob_lines[class_name] = line
    
    ax_prob.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax_prob.set_ylabel('Probability')
    ax_prob.set_ylim([0, 1])
    ax_prob.legend(loc='upper right')
    ax_prob.grid(True, alpha=0.3)
    
    # Plot spectrogram
    f, t_spec, Sxx = scipy_signal.spectrogram(trace.data, fs=sampling_rate, nperseg=256)
    im = ax_spec.pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10), 
                            shading='gouraud', cmap='viridis', vmin=-50, vmax=20)
    ax_spec.set_ylabel('Frequency (Hz)')
    ax_spec.set_ylim([0, min(50, sampling_rate/2)])
    plt.colorbar(im, ax=ax_spec, label='Power (dB)')
    
    # Plot event summary
    event_data = []
    for class_name, class_events in events.items():
        for event in class_events:
            start_time = event.get('start_time', event.get('start', 0) / sampling_rate)
            event_data.append([class_name, start_time, event.get('max_prob', 0)])
    
    if event_data:
        event_data = np.array(event_data, dtype=object)
        class_names = event_data[:, 0]
        start_times = event_data[:, 1].astype(float)
        max_probs = event_data[:, 2].astype(float)
        
        for class_name in set(class_names):
            mask = class_names == class_name
            color = colors.get(class_name, 'black')
            ax_events.scatter(start_times[mask], max_probs[mask], 
                            c=color, label=class_name.upper(), s=100, alpha=0.6)
    
    ax_events.set_xlabel('Time (s)')
    ax_events.set_ylabel('Max Probability')
    ax_events.set_ylim([0, 1])
    ax_events.legend(loc='upper right')
    ax_events.grid(True, alpha=0.3)
    
    # Add event windows as vertical spans
    event_spans = {}
    for class_name, class_events in events.items():
        color = colors.get(class_name, 'black')
        for event in class_events:
            start_t = event.get('start_time', event.get('start', 0) / sampling_rate)
            end_t = event.get('end_time', event.get('end', 0) / sampling_rate)
            
            span = ax_wave.axvspan(start_t, end_t, alpha=0.2, color=color, visible=True)
            if class_name not in event_spans:
                event_spans[class_name] = []
            event_spans[class_name].append(span)
    
    # Add checkboxes to toggle event visibility
    rax = plt.axes([0.02, 0.4, 0.10, 0.15])
    labels = list(events.keys())
    visibility = [True] * len(labels)
    check = CheckButtons(rax, labels, visibility)
    
    def toggle_events(label):
        """Toggle visibility of events for a class."""
        if label in event_spans:
            for span in event_spans[label]:
                span.set_visible(not span.get_visible())
        if label in prob_lines:
            line = prob_lines[label]
            line.set_visible(not line.get_visible())
        fig.canvas.draw_idle()
    
    check.on_clicked(toggle_events)
    
    plt.show()
    
    return fig


def plot_event_summary(
    events: Dict[str, List[Dict]],
    sampling_rate: float = 100.0,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
):
    """
    Plot summary statistics of detected events.
    
    Args:
        events (Dict[str, List[Dict]]): Detected events per class
        sampling_rate (float): Sampling rate in Hz
        figsize (tuple): Figure size
        save_path (str): Optional path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.ravel()
    
    colors = {'eq': 'red', 'px': 'blue', 'su': 'green'}
    
    # Collect data
    all_durations = {cls: [] for cls in events.keys()}
    all_max_probs = {cls: [] for cls in events.keys()}
    all_mean_probs = {cls: [] for cls in events.keys()}
    all_aucs = {cls: [] for cls in events.keys()}
    
    for class_name, class_events in events.items():
        for event in class_events:
            duration = event.get('duration_sec', event.get('duration', 0) / sampling_rate)
            all_durations[class_name].append(duration)
            all_max_probs[class_name].append(event.get('max_prob', 0))
            all_mean_probs[class_name].append(event.get('mean_prob', 0))
            all_aucs[class_name].append(event.get('area_under_curve', 0))
    
    # Plot duration distribution
    for class_name, durations in all_durations.items():
        if durations:
            color = colors.get(class_name, 'black')
            axes[0].hist(durations, bins=20, alpha=0.5, color=color, label=class_name.upper())
    axes[0].set_xlabel('Duration (s)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Event Duration Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot max probability distribution
    for class_name, max_probs in all_max_probs.items():
        if max_probs:
            color = colors.get(class_name, 'black')
            axes[1].hist(max_probs, bins=20, alpha=0.5, color=color, label=class_name.upper())
    axes[1].set_xlabel('Max Probability')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Max Probability Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot mean prob vs duration
    for class_name, mean_probs in all_mean_probs.items():
        if mean_probs:
            color = colors.get(class_name, 'black')
            durations = all_durations[class_name]
            axes[2].scatter(durations, mean_probs, c=color, alpha=0.5, 
                          label=class_name.upper(), s=50)
    axes[2].set_xlabel('Duration (s)')
    axes[2].set_ylabel('Mean Probability')
    axes[2].set_title('Mean Probability vs Duration')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Plot event count per class
    counts = [len(class_events) for class_events in events.values()]
    class_labels = list(events.keys())
    class_colors = [colors.get(cls, 'black') for cls in class_labels]
    axes[3].bar(class_labels, counts, color=class_colors, alpha=0.7)
    axes[3].set_ylabel('Count')
    axes[3].set_title('Event Count by Class')
    axes[3].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig, axes
