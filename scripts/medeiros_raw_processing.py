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
#   data needed for the deep learning models in a parquet file
#
# Usage:
#   1. Install required packages:
#       pandas
#       mne
#       scikit-learn
#   2. Adjust DATA_PATH / STORAGE_PATH / WINDOW_SIZE as needed
#   3. Run the script with:
#       python3 ./medeiros_raw_processing.py
# ------------------------------------------
import os
import mne
import pandas as pd
from sklearn.preprocessing import minmax_scale

DATA_PATH = '~/Git/Thesis/thesis_data_transformed'
STORAGE_PATH = './data/medeiros/medeiros_raw.parquet'
WINDOW_SIZE = 2000 # in milliseconds

PARTICIPANTS = ['S01', 'S03', 'S04', 'S05', 'S07', 'S08', 'S10', 'S11', 'S12',
    'S13', 'S14', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24',
    'S25', 'S26', 'S27', 'S28', 'S29', 'S30']

for participant_name in PARTICIPANTS:
    file1 = DATA_PATH + '/' + participant_name + '/' + participant_name + 'R01_an.set'
    file2 = DATA_PATH + '/' + participant_name + '/' + participant_name + 'R02_an.set'
    file3 = DATA_PATH + '/' + participant_name + '/' + participant_name + 'R03_an.set'

    raw1 = mne.io.read_raw_eeglab(file1)
    raw2 = mne.io.read_raw_eeglab(file2)
    raw3 = mne.io.read_raw_eeglab(file3)

    # Load if it was not loaded (could be if .set file has only header information)
    if raw1.preload == False:
        raw1.load_data()

    # Convert to dataframe
    temp = raw1.to_data_frame()
    temp.drop('time', axis = 1, inplace = True)

    # Extract events data from EEGLAB file
    events, _ = mne.events_from_annotations(raw1)
    raw1.close()

    # We know that the order of events is the same - see Medeiros et al. (2021) - Figure 1
    # Thus, we can just select 2nd event in the chain
    #   (i.e. reading control task) and 4th event in the chain (i.e. code comprehension task)
    #   and extract them
    timeframe_control = (events[1][0], events[2][0])
    timeframe_task = (events[3][0], events[4][0])

    # As it's in milliseconds, we can splice the dataframe instead of trying to look up by time
    control = temp.iloc[timeframe_control[0]:timeframe_control[1]]
    # Truncate to last second (we are windowing with varying window size)
    to_truncate = control.shape[0] % WINDOW_SIZE
    control.drop(control.tail(to_truncate).index, inplace = True)

    task = temp.iloc[timeframe_task[0]:timeframe_task[1]]
    # Truncate to last second (we are windowing with varying window size)
    to_truncate = task.shape[0] % WINDOW_SIZE
    task.drop(task.tail(to_truncate).index, inplace = True)

    # Add labels to both dataframes
    control['label'] = 0
    task['label'] = 1
    # Append to new dataframe
    df = pd.concat([control, task], ignore_index = True)

    if raw2.preload == False:
        raw2.load_data()
    # Repeat for other two tasks
    temp = raw2.to_data_frame()
    temp.drop('time', axis = 1, inplace = True)

    events, _ = mne.events_from_annotations(raw2)
    raw2.close()

    timeframe_control = (events[1][0], events[2][0])
    timeframe_task = (events[3][0], events[4][0])

    control = temp.iloc[timeframe_control[0]:timeframe_control[1]]
    to_truncate = control.shape[0] % WINDOW_SIZE
    control.drop(control.tail(to_truncate).index, inplace = True)

    task = temp.iloc[timeframe_task[0]:timeframe_task[1]]
    to_truncate = task.shape[0] % WINDOW_SIZE
    task.drop(task.tail(to_truncate).index, inplace = True)

    control['label'] = 0
    task['label'] = 2
    df = pd.concat([df, control, task], ignore_index = True)

    if raw3.preload == False:
        raw3.load()

    temp = raw3.to_data_frame()
    temp.drop('time', axis = 1, inplace = True)

    events, _ = mne.events_from_annotations(raw3)
    raw3.close()

    timeframe_control = (events[1][0], events[2][0])
    timeframe_task = (events[3][0], events[4][0])

    control = temp.iloc[timeframe_control[0]:timeframe_control[1]]
    to_truncate = control.shape[0] % WINDOW_SIZE
    control.drop(control.tail(to_truncate).index, inplace = True)

    task = temp.iloc[timeframe_task[0]:timeframe_task[1]]
    to_truncate = task.shape[0] % WINDOW_SIZE
    task.drop(task.tail(to_truncate).index, inplace = True)

    control['label'] = 0
    task['label'] = 3
    df = pd.concat([df, control, task], ignore_index = True)

    # Also do one-hot encoding
    one_hot = pd.get_dummies(df['label'])
    one_hot = one_hot.add_prefix('label', axis = 1)

    df.drop('label', axis = 1, inplace = True)
    df = df.join(one_hot)

    for column in df.columns:
        if column.startswith('label'):
            continue
        df[column] = minmax_scale(df[column])

    if os.path.isfile(STORAGE_PATH):
        df.to_parquet(STORAGE_PATH, engine = 'fastparquet', index = False, append = True)
    else:
        df.to_parquet(STORAGE_PATH, engine = 'fastparquet', index = False)

    # DEBUG
    if participant_name == 'S04':
        break