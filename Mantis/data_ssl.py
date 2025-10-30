import os
import numpy as np
import pandas as pd
import torch
from typing import Tuple, Optional, List
import json


def load_csv_data(file_path):
    """Load single CSV file with robust error handling"""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        data_start = 0
        for i, line in enumerate(lines):
            if "TIME" in line and "CH1" in line:
                data_start = i
                break
        
        # Read CSV from header row
        df = pd.read_csv(file_path, skiprows=data_start, header=0, dtype=str)
        
        channel_cols = ['CH1', 'CH2', 'CH3', 'CH4']
        for col in channel_cols:
            if col in df.columns:
                # Clean and convert to numeric
                df[col] = df[col].astype(str).str.replace(r'[^\d\.\-eE]', '', regex=True)
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Apply probe attenuation
                df[col] = df[col] * 10.0
        
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def parse_csv_filename(filename):
    """Parse CSV filename format: class_cSampleIdxcDrivingFreqcLoadPct.csv"""
    if not filename.endswith('.csv'):
        return None
    
    name = filename[:-4]
    parts = name.split('c')
    
    if len(parts) != 4:
        return None
    
    try:
        first_part = parts[0]
        if first_part.endswith('_'):
            class_id = int(first_part[:-1])
        else:
            class_id = int(first_part)
        
        sample_idx = int(parts[1])
        driving_freq = int(parts[2])
        load_pct = int(parts[3])
        
        class_names = {
            0: "Normal", 1: "HI-1", 2: "HI-2", 3: "HI-3",
            4: "LI-1", 5: "LI-2", 6: "LI-3"
        }
        
        return {
            'class_id': class_id,
            'class_name': class_names.get(class_id, f"Unknown_{class_id}"),
            'sample_idx': sample_idx,
            'driving_freq': driving_freq,
            'load_pct': load_pct
        }
    except (ValueError, IndexError, KeyError) as e:
        print(f"Error parsing filename {filename}: {e}")
        return None


def segment_csv_data(df, window_size=512, stride=256):
    """Segment time series data into fixed windows"""
    segments = []
    channels = ['CH1', 'CH2', 'CH3', 'CH4']
    available_channels = [ch for ch in channels if ch in df.columns]
    
    if not available_channels:
        return np.array([])
    
    data = df[available_channels].values.T  # Shape: (4, time_points)
    
    for start in range(0, data.shape[1] - window_size + 1, stride):
        segment = data[:, start:start + window_size]
        segments.append(segment)
    
    if segments:
        return np.stack(segments, axis=0)  # (n_segments, 4, window_size)
    return np.array([])


def load_csv_dataset_split(data_path, 
                           window_size=512, 
                           stride=256,
                           selected_freqs=None, 
                           selected_loads=None,
                           train_ratio=0.01):
    """
    Load CSV dataset with train/test split
    
    Args:
        data_path: Path to CSV files
        window_size: Window size for segmentation
        stride: Stride for sliding window
        selected_freqs: List of frequencies to use (None = all)
        selected_loads: List of loads to use (None = all)
        train_ratio: Ratio of data to use for training
        
    Returns:
        train_data: (N_train, 4, 512) numpy array
        train_labels: (N_train,) numpy array
        test_data: (N_test, 4, 512) numpy array
        test_labels: (N_test,) numpy array
    """
    
    print(f"Loading CSV dataset from: {data_path}")
    print(f"Window size: {window_size}, Stride: {stride}")
    print(f"Train ratio: {train_ratio}")
    
    # Scan files
    file_metas = []
    csv_files = [f for f in os.listdir(data_path) 
                 if f.endswith('.csv') and not f.startswith('._')]
    
    for filename in csv_files:
        meta = parse_csv_filename(filename)
        if meta is None:
            continue
        if selected_freqs and meta['driving_freq'] not in selected_freqs:
            continue
        if selected_loads and meta['load_pct'] not in selected_loads:
            continue
        
        file_path = os.path.join(data_path, filename)
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            data_start = 0
            for i, line in enumerate(lines):
                if "TIME" in line and "CH1" in line:
                    data_start = i
                    break
            total_points = len(lines) - data_start - 1
            if total_points <= 0:
                continue
            n_segments = max(0, (total_points - window_size) // stride + 1)
            if n_segments > 0:
                file_metas.append((file_path, meta['class_id'], n_segments))
        except Exception as e:
            print(f"Skip {filename}: {e}")
            continue
    
    if not file_metas:
        raise ValueError("No valid CSV files found")
    
    print(f"Found {len(file_metas)} valid CSV files")
    
    # Build sample index
    all_sample_indices = []
    for file_idx, (file_path, class_id, n_segs) in enumerate(file_metas):
        for seg_idx in range(n_segs):
            all_sample_indices.append((file_idx, seg_idx, class_id))
    
    total_samples = len(all_sample_indices)
    print(f"Total segments: {total_samples}")
    
    # Stratified shuffle by class
    class_groups = {}
    for item in all_sample_indices:
        cls = item[2]
        if cls not in class_groups:
            class_groups[cls] = []
        class_groups[cls].append(item)
    
    for cls in class_groups:
        np.random.shuffle(class_groups[cls])
    
    shuffled_indices = []
    max_len = max(len(v) for v in class_groups.values())
    for i in range(max_len):
        for cls in sorted(class_groups.keys()):
            if i < len(class_groups[cls]):
                shuffled_indices.append(class_groups[cls][i])
    
    # Split: 20% test, then from remaining take train_ratio for training
    test_size = int(total_samples * 0.1)
    test_indices = shuffled_indices[:test_size]
    remaining_indices = shuffled_indices[test_size:]
    
    train_pool_size = max(1, int(len(remaining_indices) * train_ratio))
    train_indices = remaining_indices[:train_pool_size]
    
    print(f"Split - Train: {len(train_indices)}, Test: {len(test_indices)}")
    
    # Load samples
    def _load_samples(sample_list):
        if not sample_list:
            return np.empty((0, 4, window_size), dtype=np.float32), np.empty((0,), dtype=np.int64)
        
        segments = []
        labels = []
        file_to_segs = {}
        for file_idx, seg_idx, class_id in sample_list:
            if file_idx not in file_to_segs:
                file_to_segs[file_idx] = []
            file_to_segs[file_idx].append((seg_idx, class_id))
        
        for file_idx, seg_info in file_to_segs.items():
            file_path = file_metas[file_idx][0]
            df = load_csv_data(file_path)
            if df is None:
                continue
            
            channel_cols = ['CH1', 'CH2', 'CH3', 'CH4']
            if not all(col in df.columns for col in channel_cols):
                continue
            
            data = df[channel_cols].values.T.astype(np.float32)  # (4, T)
            if data.shape[0] != 4:
                continue
            
            for seg_idx, class_id in seg_info:
                start = seg_idx * stride
                end = start + window_size
                if end <= data.shape[1]:
                    segment = data[:, start:end]
                    if segment.shape == (4, window_size) and np.isfinite(segment).all():
                        segments.append(segment)
                        labels.append(class_id)
        
        if segments:
            X = np.stack(segments, axis=0).astype(np.float32)
            y = np.array(labels, dtype=np.int64)
            return X, y
        else:
            return np.empty((0, 4, window_size), dtype=np.float32), np.empty((0,), dtype=np.int64)
    
    X_train, y_train = _load_samples(train_indices)
    X_test, y_test = _load_samples(test_indices)
    
    print(f"Loaded train data: {X_train.shape}, labels: {y_train.shape}")
    print(f"Loaded test data: {X_test.shape}, labels: {y_test.shape}")
    
    return X_train, y_train, X_test, y_test


def data_load(train_ratio: float = 0.01,
              csv_data_path: str = '/root/autodl-tmp/Renomeado',
              window_size: int = 512,
              stride: int = 256,
              selected_freqs: Optional[List[int]] = None,
              selected_loads: Optional[List[int]] = None):
    """
    Main data loading function compatible with Mantis fine-tuning script
    
    Returns:
        test: Tuple of (None, test_data, test_labels)
        train: Tuple of (None, train_data, train_labels)
        
    Note: Returns None for first element to match original API where
          current and voltage were separate. Here we combine all 4 channels.
    """
    
    X_train, y_train, X_test, y_test = load_csv_dataset_split(
        csv_data_path,
        window_size=window_size,
        stride=stride,
        selected_freqs=selected_freqs,
        selected_loads=selected_loads,
        train_ratio=train_ratio
    )
    
    # Mantis expects data in format (N, channels, time_steps)
    # CSV data is already in (N, 4, 512) format
    
    # Original API returned (current, voltage, labels)
    # For CSV with 4 channels, we'll treat all as "voltage" equivalent
    # and set current to None
    
    train = (None, X_train, y_train)  # (None, (N, 4, 512), (N,))
    test = (None, X_test, y_test)     # (None, (N, 4, 512), (N,))
    
    return test, train


# Additional utility for cache management
def get_cache_path(data_path, window_size, stride, train_ratio, selected_freqs, selected_loads):
    """Generate cache file path based on parameters"""
    cache_dir = os.path.join(data_path, "mantis_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    freq_str = "_".join(map(str, selected_freqs)) if selected_freqs else "all"
    load_str = "_".join(map(str, selected_loads)) if selected_loads else "all"
    
    cache_name = f"mantis_freq_{freq_str}_load_{load_str}_win_{window_size}_stride_{stride}_ratio_{train_ratio:.4f}.npz"
    return os.path.join(cache_dir, cache_name)


def data_load_with_cache(train_ratio: float = 0.01,
                         csv_data_path: str = '/root/autodl-tmp/Renomeado',
                         window_size: int = 512,
                         stride: int = 256,
                         selected_freqs: Optional[List[int]] = None,
                         selected_loads: Optional[List[int]] = None,
                         force_rebuild: bool = False):
    """
    Data loading with caching for faster subsequent loads
    """
    
    cache_path = get_cache_path(csv_data_path, window_size, stride, train_ratio, 
                                selected_freqs, selected_loads)
    
    if not force_rebuild and os.path.exists(cache_path):
        print(f"Loading cached data from: {cache_path}")
        cached = np.load(cache_path)
        X_train = cached['X_train']
        y_train = cached['y_train']
        X_test = cached['X_test']
        y_test = cached['y_test']
    else:
        print("Building dataset from scratch...")
        X_train, y_train, X_test, y_test = load_csv_dataset_split(
            csv_data_path, window_size, stride,
            selected_freqs, selected_loads, train_ratio
        )
        
        # Save to cache
        np.savez(cache_path,
                 X_train=X_train, y_train=y_train,
                 X_test=X_test, y_test=y_test)
        print(f"Cached data saved to: {cache_path}")
    
    train = (None, X_train, y_train)
    test = (None, X_test, y_test)
    
    return test, train