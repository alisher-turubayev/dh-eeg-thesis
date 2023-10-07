# Automated Machine Learning data preparation script for BASE Mental
#   Effort Monitoring Dataset by Medeiros et al (2021)
#
# Full reference:
#
# Medeiros, J., Couceiro, R., Duarte, G., Durães, J., Castelhano, J.,
#   Duarte, C., Castelo-Branco, M., Madeira, H., de Carvalho, P., &
#   Teixeira, C. (2021). Can EEG Be Adopted as a Neuroscience Reference
#   for Assessing Software Programmers' Cognitive Load? Sensors, 21(7),
#   2338. https://doi.org/10.3390/s21072338
#
# Work completed as part of the Master's Thesis for M.Sc. Digital Health
#   @ Hasso-Plattner Institute, University of Potsdam, Germany
#
# Authors: Alisher Turubayev, Fabian Stolp (PhD supervisor)
#
# Goals:
#   open .set files in a specified directory, do feature extraction,
#   normalization, and transformation, and pack the data into parquet
#   file(s)
#
# Usage:
#   1. Install required packages:
#       pandas
#       mne
#       scikit-learn
#       antropy
#       scipy
#   2. Adjust DATA_PATH / STORAGE_PATH as needed
#   3. Run the script with:
#       python3 ./medeiros_raw_processing.py
#
# Notes:
#   1. Mean of normalized signal was done as mean of min-max scaled signal
#   2. The number of power ratio features is increased to 42, 
#       as there are 7 * 6 pairs of frequency bands. In total, there are 68
#       uni-channel features instead of declared 48 in the paper.
#   3. The number of differential and absolute asymmetry features is increased to 
#       168 from 126. We could not understand what 9 pairs were used for the calculation,
#       so we opted to do these calculations with the 12 pairs described in 
#       Lin et al. (2010) http://ieeexplore.ieee.org/document/5458075/.
# ------------------------------------------
import os
from datetime import datetime
import mne
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("error")

from ml_utils import extract_features, normalize_val, split_into_segments

DATA_PATH = '~/data/medeiros_processed_extracted'
STORAGE_PATH = './data/medeiros/medeiros_processed'

# Participant IDs
PARTICIPANTS = ['S01', 'S03', 'S04', 'S05', 'S07', 'S08', 'S10', 'S11', 'S12',
    'S13', 'S14', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24',
    'S25', 'S26', 'S27', 'S28', 'S29', 'S30']

# File associations
FILE_SUFFIXES = {
    'task1': 'R01', 
    'task2': 'R02', 
    'task3': 'R03'
    }
# One-hot encoding for labels
TASK_CODES = {
    'control': [1, 0, 0, 0], 
    'task1': [0, 1, 0, 0], 
    'task2': [0, 0, 1, 0], 
    'task3': [0, 0, 0, 1]
    }
TASK_COL_LABELS = ['sample_index', 'label0', 'label1', 'label2', 'label3']

start_time = datetime.now()

# Expand the user ~ to absolute path - needed so that the os.path.isfile check works
if DATA_PATH[0] == '~':
    DATA_PATH = os.path.expanduser(DATA_PATH)

# Make sure storage path exists
os.makedirs(STORAGE_PATH, exist_ok = True)

# Remove files where to store if exists (we are overriding it)
for file in os.listdir(STORAGE_PATH):
    os.remove(os.path.join(STORAGE_PATH, file))

for participant_name in PARTICIPANTS:
    print(f'Processing participant {participant_name}')
    participant_start_time = datetime.now()

    # Need a grouped control for feature transformation as in Medeiros et al. (2021) p.12
    grouped_control_features = pd.DataFrame()

    processed_files = 0
    for task_key in FILE_SUFFIXES:
        file_name = participant_name + FILE_SUFFIXES[task_key] + '_an.set'
        full_path = os.path.join(DATA_PATH, participant_name, file_name)

        print(f'  Processing file {file_name}...')

        if os.path.isfile(full_path) == False:
            print(f'  No file {full_path} was found. Continuing...')
            break

        # Preload file
        raw = mne.io.read_raw_eeglab(full_path, preload = True, verbose = False)
        # Convert to dataframe and drop unneeded time column 
        #   (the converted dataframe is time-synced anyway)
        df = raw.to_data_frame() 
        df.drop('time', axis = 1, inplace = True)
        # Extract events information from EEGLAB (timestamps of events)
        events, _ = mne.events_from_annotations(raw, verbose = False)
        # Important to close raw file
        raw.close()

        # We know that the order of events is the same - see Medeiros et al. (2021) - Figure 1
        # Thus, we can just select all subsections that are of interest for the scirpting
        #   from the beginning of dataset to end of comprehension task
        timeframe_control_baseline = (events[0][0], events[1][0])
        timeframe_control = (events[1][0], events[2][0])
        timeframe_task_baseline = (events[2][0], events[3][0])
        timeframe_task = (events[3][0], events[4][0])

        #### Feature Extraction
        # Extract subsections and extract features of interest
        control_baseline_features = extract_features(df.iloc[timeframe_control_baseline[0]:timeframe_control_baseline[1]])
        control_features = extract_features(df.iloc[timeframe_control[0]:timeframe_control[1]])
        task_baseline_features = extract_features(df.iloc[timeframe_task_baseline[0]:timeframe_task_baseline[1]])
        task_features = extract_features(df.iloc[timeframe_task[0]:timeframe_task[1]])
        del df

        #### Feature Normalization
        for column in control_baseline_features.columns:
            column_mean = control_baseline_features[column].mean()
            control_features[column] = control_features[column].apply(lambda x: normalize_val(x, column_mean))

        for column in task_baseline_features.columns:
            column_mean = task_baseline_features[column].mean()
            task_features[column] = task_features[column].apply(lambda x: normalize_val(x, column_mean))

        # Known to produce 0s because power can be close to 0 and therefore eval to 0 in some cases
        # Floating point shenanigans :3
        control_features = control_features.fillna(0.0)
        task_features = task_features.fillna(0.0)

        grouped_control_features = pd.concat([grouped_control_features, control_features], ignore_index = True)
        #### Feature Transformation
        # Note that feature transformation for the control task takes place outside of this for loop
        task = pd.DataFrame()
        segments = split_into_segments(task_features)
        for segment in segments:
            seg_max = segment.max(skipna = True, numeric_only = True).to_frame().T
            seg_max = seg_max.add_prefix('max_')
            seg_min = segment.min(skipna = True, numeric_only = True).to_frame().T
            seg_min = seg_min.add_prefix('min_')
            seg_mean = segment.mean(skipna = True, numeric_only = True).to_frame().T
            seg_mean = seg_mean.add_prefix('mean_')
            seg_std = segment.std(skipna = True, numeric_only = True).to_frame().T
            seg_std = seg_std.add_prefix('std_')
            seg_median = segment.median(skipna = True, numeric_only = True).to_frame().T
            seg_median = seg_median.add_prefix('median_')
            segment_agg = pd.concat([seg_max, seg_min, seg_mean, seg_std, seg_median], axis = 1)
            task = pd.concat([task, segment_agg], ignore_index = True)
        # Add labels
        task['label0'] = 0
        task['label1'] = TASK_CODES[task_key][1]
        task['label2'] = TASK_CODES[task_key][2]
        task['label3'] = TASK_CODES[task_key][3]

        task.to_parquet(os.path.join(STORAGE_PATH, participant_name + FILE_SUFFIXES[task_key] + '.parquet'))
        processed_files += 1

    if processed_files == 0:
        continue
    # Group as in Medeiros et al. (2021)
    control = pd.DataFrame()  
    segments = split_into_segments(grouped_control_features)
    for segment in segments:
        seg_max = segment.max(skipna = True, numeric_only = True).to_frame().T
        seg_max = seg_max.add_prefix('max_')
        seg_min = segment.min(skipna = True, numeric_only = True).to_frame().T
        seg_min = seg_min.add_prefix('min_')
        seg_mean = segment.mean(skipna = True, numeric_only = True).to_frame().T
        seg_mean = seg_mean.add_prefix('mean_')
        seg_std = segment.std(skipna = True, numeric_only = True).to_frame().T
        seg_std = seg_std.add_prefix('std_')
        seg_median = segment.median(skipna = True, numeric_only = True).to_frame().T
        seg_median = seg_median.add_prefix('median_')
        segment_agg = pd.concat([seg_max, seg_min, seg_mean, seg_std, seg_median], axis = 1)
        control = pd.concat([task, segment_agg], ignore_index = True)
    # Add labels
    control['label0'] = 1
    control['label1'] = 0
    control['label2'] = 0
    control['label3'] = 0

    control.to_parquet(os.path.join(STORAGE_PATH, participant_name + '_control.parquet'))
    
    participant_end_time = datetime.now()
    print(f'-> Processing for participant {participant_name} finished. Time to execute: {participant_end_time - participant_start_time}')

end_time = datetime.now()

print('')
print(f'Finished data processing. Total time to execute: {end_time - start_time}')