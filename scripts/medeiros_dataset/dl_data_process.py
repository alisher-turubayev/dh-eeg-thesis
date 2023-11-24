# Automated Deep Learning data preparation script for BASE Mental Effort Monitoring 
#   Dataset by Medeiros et al (2021)
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
#   open .set files in a specified directory, truncate unneeded data and store
#   data needed for the deep learning models in a parquet files
#
# Usage:
#   1. Install required packages:
#       pandas
#       mne
#       scikit-learn
#   2. Adjust DATA_PATH / STORAGE_PATH / WINDOW_SIZE as needed
#   3. Run the script with:
#       python3 ./medeiros_raw_processing.py
#
# NOTES:
# During initial development, a script with a single file .parquet storage was 
#   created. However, due to big size of the resultant file (~11 GB) it was not
#   feasable to continue using it for DL training. As a result, this file was 
#   rewritten to store data in separate files. Adjustments to the dataloader.py 
#   file were also made to accommodate this change. 
# ------------------------------------------
import os
import mne
from datetime import datetime
import pandas as pd

DATA_PATH = '~/data/medeiros_raw_extracted'
STORAGE_PATH = './data/medeiros/medeiros_raw'
METADATA_FILEPATH = os.path.join(STORAGE_PATH, 'metadata.csv')
WINDOW_SIZE = 2000 # in milliseconds

PARTICIPANTS = ['S01', 'S03', 'S04', 'S05', 'S07', 'S08', 'S09', 'S10', 'S11', 
    'S12', 'S13', 'S14', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 
    'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30']
# File associations
FILE_SUFFIXES = {
    'task1': 'R01_an.set', 
    'task2': 'R02_an.set', 
    'task3': 'R03_an.set'
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

# Initialize index variable - this will help in storing the dataset 
#   in separate files
sample_index = 0

metadata = dict()
for participant_name in PARTICIPANTS:
    print(f'Processing participant {participant_name}')

    for task_key in FILE_SUFFIXES:
        file_name = participant_name + FILE_SUFFIXES[task_key]
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
        # Thus, we can just select 2nd event in the chain
        #   (i.e. reading control task) and 4th event in the chain (i.e. code comprehension task)
        #   and extract them
        timeframe_control = (events[1][0], events[2][0])
        timeframe_task = (events[3][0], events[4][0])

        # As it's in milliseconds, we can splice the dataframe instead of trying to look up by time
        control = df.iloc[timeframe_control[0]:timeframe_control[1]]
        task = df.iloc[timeframe_task[0]:timeframe_task[1]]
        
        # Extract samples
        samples = [control.iloc[int(i * WINDOW_SIZE):int((i + 1) * WINDOW_SIZE)] for i in range(int(control.shape[0] / WINDOW_SIZE))]
        # Store each sample in a separate file - see NOTES
        for sample in samples:
            # Create a new file
            sample.to_parquet(os.path.join(STORAGE_PATH, str(sample_index) + '.parquet'))
            # Add metadata about this file
            row_to_insert = [sample_index]
            row_to_insert.extend(TASK_CODES['control'])
            metadata[sample_index] = row_to_insert
            sample_index += 1
            
        # Repeat for task data
        samples = [task.iloc[int(i * WINDOW_SIZE):int((i + 1) * WINDOW_SIZE)] for i in range(int(task.shape[0] / WINDOW_SIZE))]
        for sample in samples:
            sample.to_parquet(os.path.join(STORAGE_PATH, str(sample_index) + '.parquet'))
            row_to_insert = [sample_index]
            row_to_insert.extend(TASK_CODES[task_key])
            metadata[sample_index] = row_to_insert
            sample_index += 1

metadata_df = pd.DataFrame.from_dict(metadata, orient = 'index', columns = TASK_COL_LABELS)
metadata_df.to_csv(METADATA_FILEPATH)

end_time = datetime.now()

print(f'Finished data processing. Total time to execute: {end_time - start_time}, total samples processed: {sample_index}')