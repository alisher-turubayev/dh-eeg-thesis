# Import required libraries
import mne

# For batch processing
import glob

# For time measurement
import time

# Set path to data (with pattern matching)
data_path = r'../Thesis/thesis_data_transformed/**/*.set'

# Get the list of items to process
file_list = glob.glob(data_path, recursive = True)
print(f'{len(file_list)} items to process found.')

# Set four channel types (HEOG, VEOG, EMG, ECG) to non-EEG signal for setting montage later
chan_types = {'HEO': 'eog', 'VEO': 'eog', 'EMG': 'emg', 'EKG': 'ecg'}

start_time = time.time()

for path in file_list:
    print(f'Processing file {path}...')
    # Load the file in
    raw_data = mne.io.read_raw_eeglab(path)
    # Change channel types and set montage (channel location information)
    raw_data.set_channel_types(chan_types)
    raw_data.set_montage('standard_1020', match_alias = True, match_case = False)
    # Apply filtering (1 Hz / 90 Hz / 50 Hz notch)
    raw_data.filter(l_freq = 1, h_freq = 90)
    raw_data.notch_filter(freqs = 50)
    # Export back into the EEGLAB folmat for now - this is going to be removed after bad channels are labelled as such
    mne.export.export_raw(path, raw_data, fmt = 'eeglab', overwrite = True)
    print(f'File {path} processed and exported\n')
    
    # TODO:
    # 1. Interpolation of bad channels
    # 2. ICA automatic component removal (there is a way to do in within MNE - but not sure if only for ocular artifacts)
    # 3. Group analysis (how to do so?)

end_time = time.time()

print(f'File processing finished. Total time: {end_time - start_time} seconds.')