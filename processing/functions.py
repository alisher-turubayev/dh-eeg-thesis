"""
Support functions for preprocessing.
"""
import itertools
import pandas as pd
import numpy as np
import mne

def unpack_medeiros(data: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Unpacking function for Medeiros dataset
    """
    eeg_raw = data['data'][0, 2]['data'][0, 0]
    eeg_chan_info = [ch[0] for ch in data['data']
                        [0, 2]['signals'][0, 0]['labels'][0]]
    eeg_events = data['data'][1, 2]['event'][0, 0].flatten()
    eeg_events = np.reshape(
        np.fromiter(itertools.chain.from_iterable(eeg_events), float), (6, 3))
    eeg_events_table = pd.DataFrame(eeg_events, columns = ['type', 'latency', 'urevent'])

    return eeg_raw, eeg_chan_info, eeg_events_table

def preprocess_medeiros(
        raw: mne.io.BaseRaw,
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

def segment_medeiros():
    """
    Applies segmentation into 4 segments like in Medeiros et al. (2021)
    """
    return
