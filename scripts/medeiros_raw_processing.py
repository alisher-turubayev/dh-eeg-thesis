import mne
import pandas as pd
from sklearn.preprocessing import minmax_scale

DATA_PATH = '~/Git/Thesis/thesis_data_transformed/S30/'
STORAGE_PATH = '../data/medeiros/medeiros_raw.parquet'
WINDOW_SIZE = 2000 # in milliseconds

raw1 = mne.io.read_raw_eeglab(DATA_PATH + 'S30R01_an.set')
raw2 = mne.io.read_raw_eeglab(DATA_PATH + 'S30R02_an.set')
raw3 = mne.io.read_raw_eeglab(DATA_PATH + 'S30R03_an.set')

# Convert to dataframe
temp = raw1.to_data_frame()
temp.drop('time', axis = 1, inplace = True)

# Extract events data from EEGLAB file
events, _ = mne.events_from_annotations(raw1)
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

# Repeat for other two tasks
temp = raw2.to_data_frame()
temp.drop('time', axis = 1, inplace = True)

events, _ = mne.events_from_annotations(raw2)
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

temp = raw3.to_data_frame()
temp.drop('time', axis = 1, inplace = True)

events, _ = mne.events_from_annotations(raw3)
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

df.to_parquet(STORAGE_PATH, engine = 'fastparquet', index = False, )