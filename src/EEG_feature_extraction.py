import os
import glob
import numpy as np
import re
import pandas as pd
import hfd # reference to https://github.com/inuritdino/HiguchiFractalDimension
from scipy.signal import butter, lfilter, welch
from scipy.stats import skew, kurtosis

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)

# Define frequency bands.
band_dict = {
    'delta': [1, 4],
    'theta': [4, 8],
    'alpha': [8, 12],
    'beta':  [12, 30],
    'gamma': [30, 45]
}

# Load the CSV file containing participant information.
csv_data = pd.read_csv('participants.csv', sep='\t')

skipped_log = []
fs = 256
id_pattern = re.compile(r'(sub-\d+)')
npz_files = glob.glob('data/EEG_data/resting_state_preprocessed_data_new/*.npz')

# Create an output directory for the features.
output_dir = 'data/EEG_data/HFD_PSD_4sec_75overlap_6min'
os.makedirs(output_dir, exist_ok=True)

for npz_file in npz_files:
    print(f"Processing {npz_file}")
    base_name = os.path.basename(npz_file).split('.')[0]
    match = id_pattern.search(base_name)
    if not match:
        skipped_log.append(f"Could not extract subject id from {base_name}. Skipping file.")
        continue
    csv_subject_id = match.group(1)
    
    try:
        subject_number = int(csv_subject_id.split('-')[1])
    except ValueError:
        skipped_log.append(f"Could not convert subject number from {csv_subject_id}. Skipping file.")
        continue

    # Extract CSV information for the subject.
    subject_rows = csv_data.loc[csv_data['participant_id'] == csv_subject_id]
    if subject_rows.empty:
        skipped_log.append(f"No CSV entry for {csv_subject_id}. Skipping file.")
        continue
    csv_info = subject_rows.iloc[0].to_dict()

    print(f"Processing participant {csv_subject_id}")
    data = np.load(npz_file)
    # Scale signal as needed.
    signal = data['data'] * 1e6  
    npz_label = data['label']

    num_channels, signal_length = signal.shape

    # Trim signal to exactly 6 minutes (360 seconds)
    max_duration_seconds = 360  # 6 minutes
    max_samples = max_duration_seconds * fs  # 360 * 256 = 92160 samples
    
    if signal_length < max_samples:
        skipped_log.append(f"{csv_subject_id}: Signal too short ({signal_length/fs:.2f}s < 360s). Skipping.")
        print(f"Warning: Signal too short for {csv_subject_id}. Skipping.")
        continue
    
    # Truncate to exactly 6 minutes
    signal = signal[:, :max_samples]
    signal_length = max_samples

    # Define window size and step for 75% overlap
    window_seconds = 4
    window_size = int(window_seconds * fs)  # 4 seconds = 1024 samples at 256 Hz
    step_size = int(window_size * 0.25)  # 75% overlap = 25% step = 256 samples
    
    # Calculate number of windows with overlap
    num_windows = (signal_length - window_size) // step_size + 1
    
    print(f"Signal length: {signal_length} samples ({signal_length/fs:.2f} seconds)")
    print(f"Window size: {window_size} samples ({window_seconds} seconds)")
    print(f"Step size: {step_size} samples ({step_size/fs:.2f} seconds, 75% overlap)")
    print(f"Number of windows: {num_windows}")

    # Lists to store features for each window.
    subject_HFD_features = []          # HFD features: (num_windows, num_channels, num_bands)
    subject_PSD_features = []          # PSD features: (num_windows, num_channels, num_bands)
    subject_band_stats_features = []   # Band-specific time domain stats: (num_windows, num_channels, num_bands, 5)
    subject_raw_stats_features = []    # Raw time domain stats per channel: (num_windows, num_channels, 5)

    # Process each window with overlap.
    for w in range(num_windows):
        window_start = w * step_size
        window_end = window_start + window_size
        
        # Safety check to ensure we don't exceed signal length
        if window_end > signal_length:
            print(f"Warning: Window {w} exceeds signal length. Skipping.")
            break
            
        window_signal = signal[:, window_start:window_end]  # shape: (num_channels, window_size)

        window_HFD = []
        window_PSD = []
        window_band_stats = []
        window_raw_stats = []
        
        # Process each channel in the window.
        for ch in range(num_channels):
            channel_signal = window_signal[ch]

            # Initialize arrays for the frequency bands.
            num_bands = len(band_dict)
            hfd_features = np.zeros(num_bands)
            psd_features = np.zeros(num_bands)
            band_stats = np.zeros((num_bands, 5))  # 5 stats: mean, std, var, skew, kurtosis

            # For each frequency band, compute HFD, PSD, and time domain stats on the filtered signal.
            for i, (band_name, band_range) in enumerate(band_dict.items()):
                band_signal = butter_bandpass_filter(channel_signal, band_range[0], band_range[1], fs)
                
                # Compute HFD for the band-filtered signal.
                hfd_features[i] = hfd.hfd(band_signal, kmax=82)
                
                # Compute PSD using Welch's method.
                freqs, psd = welch(band_signal, fs=fs, nperseg=fs*2)
                # Sum PSD values within the frequency band.
                psd_features[i] = np.sum(psd[(freqs >= band_range[0]) & (freqs < band_range[1])])
                
                # Compute time domain stats for the band-filtered signal.
                band_mean = np.mean(band_signal)
                band_std = np.std(band_signal)
                band_var = np.var(band_signal)
                band_skew = skew(band_signal)
                band_kurt = kurtosis(band_signal)
                band_stats[i, :] = [band_mean, band_std, band_var, band_skew, band_kurt]
            
            # Compute overall time-domain stats for the raw channel signal.
            raw_mean = np.mean(channel_signal)
            raw_std = np.std(channel_signal)
            raw_var = np.var(channel_signal)
            raw_skew = skew(channel_signal)
            raw_kurt = kurtosis(channel_signal)
            raw_stats_features = np.array([raw_mean, raw_std, raw_var, raw_skew, raw_kurt])
            
            window_HFD.append(hfd_features)
            window_PSD.append(psd_features)
            window_band_stats.append(band_stats)
            window_raw_stats.append(raw_stats_features)
        
        subject_HFD_features.append(window_HFD)
        subject_PSD_features.append(window_PSD)
        subject_band_stats_features.append(window_band_stats)
        subject_raw_stats_features.append(window_raw_stats)

    # Update num_windows based on actual windows processed
    num_windows = len(subject_HFD_features)
    print(f"Actually processed {num_windows} windows")

    # Convert lists to numpy arrays.
    subject_HFD_features = np.array(subject_HFD_features)               # (num_windows, num_channels, num_bands)
    subject_PSD_features = np.array(subject_PSD_features)               # (num_windows, num_channels, num_bands)
    subject_band_stats_features = np.array(subject_band_stats_features) # (num_windows, num_channels, num_bands, 5)
    subject_raw_stats_features = np.array(subject_raw_stats_features)   # (num_windows, num_channels, 5)

    # Create a label array for each window.
    label_array = np.full((num_windows,), npz_label)

    # Save the computed features along with CSV information.
    output_file_path = os.path.join(output_dir, f"{subject_number}.npz")
    np.savez_compressed(output_file_path,
                        HFD_features=subject_HFD_features,
                        PSD_features=subject_PSD_features,
                        # band_stats_features=subject_band_stats_features,
                        # raw_stats_features=subject_raw_stats_features,
                        label=label_array,
                        csv_info=csv_info,
                        window_info={'window_size': window_size, 
                                   'step_size': step_size, 
                                   'overlap_percent': 75,
                                   'num_windows': num_windows,
                                   'total_duration_seconds': max_duration_seconds})
    print(f"Saved features to {output_file_path}")

# Log any files that were skipped.
if skipped_log:
    with open("skipped_log_4sec_75overlap_6min.txt", "w") as log_file:
        log_file.write("\n".join(skipped_log))