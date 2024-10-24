"""
Functions for automated data processing of the BASE Mental Effort 
Monitoring Dataset by Medeiros et al (2021)

Full reference:

Medeiros, J., Couceiro, R., Duarte, G., Durães, J., Castelhano, J.,
Duarte, C., Castelo-Branco, M., Madeira, H., de Carvalho, P., &
Teixeira, C. (2021). Can EEG Be Adopted as a Neuroscience Reference
for Assessing Software Programmers' Cognitive Load? Sensors, 21(7),
2338. https://doi.org/10.3390/s21072338

Work completed as part of the Master's Thesis for M.Sc. Digital Health
@ Hasso-Plattner Institute, University of Potsdam, Germany

Authors: Alisher Turubayev, Fabian Stolp (PhD supervisor)
"""
import os
import itertools
import pandas as pd
import numpy as np
import mne
from processing.medeiros.utils import extract_features, normalize_val, split_into_segments

def unpack_medeiros(data: dict) -> mne.io.Raw:
    # pylint: disable=not-callable
    """
    Unpacking function for Medeiros dataset
    """
    # Extract raw data (mV measurements) - in matlab, data{1,3}.data
    eeg_raw = data['data'][0, 2]['data'][0, 0]
    # Extract channel info - in matlab, data{1,3}.signals.labels column
    eeg_chan_info = [ch[0] for ch in data['data']
                        [0, 2]['signals'][0, 0]['labels'][0]]
    # Extract events information - in matlab, data{2,3}.event
    eeg_events = data['data'][1, 2]['event'][0, 0].flatten()
    # Extract sample rate - in matlab, data{1,3}.sampleRate
    sample_freq = data['data'][0, 2]['sampleRate'][0, 0][0][0]
    # Extract subject ID/condition information
    subject_id = int(data['data'][1, 3]['Run'][0, 0][0, 1][0])
    condition_id = data['data'][1, 3]['Run'][0, 0][1, 1][0]

    # Format into a dataframe so it can be added to annotations easily
    eeg_events = np.reshape(
        np.fromiter(itertools.chain.from_iterable(eeg_events), float), (6, 3))
    eeg_events_table = pd.DataFrame(eeg_events, columns = ['type', 'latency', 'urevent'])

    # Create info structure needed for MNE
    eeg_info = mne.create_info(ch_names = eeg_chan_info, sfreq = sample_freq, ch_types = 'eeg')
    # We are storing subject ID in the integer part and the condition in the string part of the ID
    eeg_info['subject_info'] = {'id': subject_id, 'hid': condition_id}

    # Create RawArray object
    raw = mne.io.RawArray(eeg_raw, eeg_info)

    # Create Annotations object
    annotations = mne.Annotations(
        onset = eeg_events_table['latency'],
        duration = 0,
        description = eeg_events_table['type']
    )
    raw.set_annotations(annotations)

    return raw

def preprocess_medeiros(
        raw: mne.io.Raw,
        remove_chans: list[str] = None,
        l_freq: float = 1.,
        h_freq: float = 90.,
        notch_freq: float = 50.):
    """
    Preprocessing function for Medeiros dataset. Replicates (as faithfully as possible)
    preprocessing steps done in Medeiros et al. (2021)
    """
    # By default, remove channels as specified by Medeiros et al. (2021)
    if remove_chans is None:
        remove_chans = ['M1', 'M2', 'CB1', 'CB2', 'HEO', 'VEO', 'EKG', 'EMG']

    # 1. Drop channels
    raw.drop_channels(remove_chans, on_missing = 'warn')
    # 2. Add channel locations
    raw.set_montage(mne.channels.make_standard_montage('standard_1020'), match_case = False)
    # 3. Apply lowpass/highpass filters
    raw = raw.filter(l_freq, h_freq)
    # 4. Apply notch filter
    raw = raw.notch_filter(notch_freq)
    # 5. Remove flat channels
    # NOTE: ignored for now
    # 6. Mark bad channels using LOF filtering
    noisy_chans = mne.preprocessing.find_bad_channels_lof(raw, picks = 'eeg')
    raw.info['bads'] = noisy_chans
    # 7. Interpolate bad channels
    raw.interpolate_bads()
    # 8. Re-reference to average
    raw = raw.set_eeg_reference(ch_type = 'eeg')
    # 9. ICA analysis with artifact removal
    ica = mne.preprocessing.ICA() # pylint: disable=not-callable
    ica.fit(raw)
    ica_idx, _ = ica.find_bads_muscle(raw)
    ica.exclude = ica_idx
    ica.apply(raw)

def pack_medeiros(files: list[os.PathLike]) -> pd.DataFrame:
    """
    Pack function for Medeiros et al. (2021) dataset
    """
    task = pd.DataFrame()
    control = pd.DataFrame()

    for file_path in files:
        # Preload file
        raw = mne.io.read_raw_eeglab(file_path, preload = True, verbose = False)
        
        # Convert to dataframe and drop unneeded time column
        #   (the converted dataframe is time-synced anyway)
        df = raw.to_data_frame()
        df.drop('time', axis = 1, inplace = True)
        # Extract events information from EEGLAB (timestamps of events)
        events, _ = mne.events_from_annotations(raw, verbose = False)
        # Extract task key from Raw.Info object
        task_key = raw.info['subject_info']['hid']
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
        control_baseline_features = extract_features(
            df.iloc[timeframe_control_baseline[0]:timeframe_control_baseline[1]]
        )
        control_features = extract_features(
            df.iloc[timeframe_control[0]:timeframe_control[1]]
        )
        task_baseline_features = extract_features(
            df.iloc[timeframe_task_baseline[0]:timeframe_task_baseline[1]]
        )
        task_features = extract_features(
            df.iloc[timeframe_task[0]:timeframe_task[1]]
        )
        del df

        #### Feature Normalization
        for column in control_baseline_features.columns:
            column_mean = control_baseline_features[column].mean()
            control_features[column] = control_features[column].apply(
                lambda x: normalize_val(x, column_mean) #pylint: disable=cell-var-from-loop
            )

        for column in task_baseline_features.columns:
            column_mean = task_baseline_features[column].mean()
            task_features[column] = task_features[column].apply(
                lambda x: normalize_val(x, column_mean) #pylint: disable=cell-var-from-loop
            )

        grouped_control_features = pd.concat(
            [grouped_control_features, control_features], ignore_index = True
        )
        #### Feature Transformation
        # Note that feature transformation for the control task takes place outside of this for loop
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
            segment_agg['label'] = task_key
            task = pd.concat([task, segment_agg], ignore_index = True)

    # Group as in Medeiros et al. (2021)
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
        segment_agg['label'] = 'control'
        control = pd.concat([control, segment_agg], ignore_index = True)

    return pd.concat([task, control], ignore_index = True)
