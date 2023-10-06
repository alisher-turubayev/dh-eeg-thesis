# Utility functions for automated data processing of the BASE Mental
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
# ------------------------------------------
import antropy as ant
import pandas as pd
import numpy as np
import pandas as pd
import scipy.stats as scs
import scipy.signal as sig
from scipy.integrate import simpson

# Electrode pairs as defined in Lin et al. (2010) http://ieeexplore.ieee.org/document/5458075/
ELECTRODE_PAIRS = [('fp1', 'fp2'),
    ('f7', 'f8'),
    ('f3', 'f4'),
    ('ft7', 'ft8'),
    ('fc3', 'fc4'),
    ('t7', 't8'),
    ('p7', 'p8'),
    ('c3', 'c4'),
    ('tp7', 'tp8'),
    ('cp3', 'cp4'),
    ('p3', 'p4'),
    ('o1', 'o2')]

SAMPLE_RATE = 1000 # in Hz
WINDOW_SIZE = 1 # in seconds
OVERLAP_RATIO = 0.5 # from 0 to 1
N_SEGMENTS = 4 # as is in Medeiros et al. (2021)

def normalize_val(x, x_hat):
    """
    Normalize value with the formula:
    (`x` - `x_hat`) / `x_hat`

    See Medeiros et al. (2021) p. 12
    """
    return (x - x_hat) / x_hat

def split_into_segments(df):
    """
    Split a `pandas.DataFrame` into segments. The number of segments is defined by `N_SEGMENTS`
    and is set to 4 by default.

    See Medeiros et al. (2021) p. 12
    """
    nperseg = int(df.shape[0] / N_SEGMENTS)
    
    segment_start = 0
    segments = []
    for _ in range(N_SEGMENTS):
        if segment_start + nperseg > df.shape[0]:
            segments.append(df.iloc[segment_start:df.shape[0]])
        else: 
            segments.append(df.iloc[segment_start:segment_start + nperseg])
            segment_start += nperseg

    return segments

# Implementation from:
# https://stackoverflow.com/a/56487241
def meanfreq(x, fs):
    """
    Returns a mean frequency estimation of the signal `x`

    See https://stackoverflow.com/a/56487241
    """
    f, Pxx_den = sig.periodogram(x, fs)                                                    
    Pxx_den = np.reshape(Pxx_den, (1, -1)) 
    width = np.tile(f[1] - f[0], (1, Pxx_den.shape[1]))
    f = np.reshape(f, (1, -1))
    P = Pxx_den * width
    pwr = np.sum(P)

    mnfreq = np.dot(P, f.T) / pwr

    return mnfreq.item()

def assemble_windows_array(s, w_size, ovlp):
    """
    Creates a list of windows with size `w_size` and overlap of `ovlp` elements from splicable `s`

    Parameters:
    `s`: array-like of elements to slice into windows
    `w_size`: size of the window in # of elements
    `ovlp`: size of the overlap in # of elements

    Returns:
    `windows`: a list of windows of size `w_size`
    """
    windows = []
    
    lb = 0
    rb = w_size

    while True:
        if rb > len(s):
            windows.append(s[lb:len(s)])
            break
        windows.append(s[lb:rb])
        lb += ovlp
        rb += ovlp

    return windows

def extract_features(df_unprocessed):
    """
    Extract all features as in Medeiros et al. (2021) pp. 10-12

    Notes: 
    1. Mean of normalized signal was done as mean of min-max scaled signal
    2. The number of power ratio features is increased to 42, 
        as there are 7 * 6 pairs of frequency bands. In total, there are 68
        uni-channel features instead of declared 48 in the paper.
    3. The number of differential and absolute asymmetry features is increased to 
        168 from 126. We could not understand what 9 pairs were used for the calculation,
        so we opted to do these calculations with the 12 pairs described in 
        Lin et al. (2010) http://ieeexplore.ieee.org/document/5458075/.
    """
    # All unichannel features of all channels across section (baseline/control/task)
    df_uf = pd.DataFrame()
    # All multichannel features of all channels across section (baseline/control_task)
    df_mf = pd.DataFrame()

    # Calculate n_per_segments & overlap size
    nperseg = WINDOW_SIZE * SAMPLE_RATE
    overlap = int(nperseg * OVERLAP_RATIO)

    # Uni-channel features
    for chan_name in df_unprocessed.columns:
        chan_windows = assemble_windows_array(df_unprocessed[chan_name], nperseg, overlap)

        df_chan_uf = pd.DataFrame()
        for window in chan_windows:
            f_window = {}
            ## Time-Domain Uni-Channel
            f_window['f_mean'] = np.mean(window)
            f_window['f_norm_mean'] = np.mean((window - np.min(window)) / (np.max(window) - np.min(window)))
            f_window['f_var'] = np.var(window)
            f_window['f_scew'] = scs.skew(window)
            f_window['f_kurt'] = scs.kurtosis(window)
            f_window['f_mob'], f_window['f_comp'] = ant.hjorth_params(window)

            ## Frequency-Domain Uni-Channel
            # fs - signal frequency, 
            freqs, psd = sig.welch(window, fs = SAMPLE_RATE, nperseg = min(nperseg, len(window)))
            # Frequencies of interest from 0 Hz to 90 Hz (p. 11)
            idx_interest = np.logical_and(freqs >= 0, freqs <= 90)
            # Frequency resolution
            freqs_res = freqs[1] - freqs[0]
            # Total power
            f_window['f_total_power'] = simpson(psd[idx_interest], dx = freqs_res)
            # Find freqs of interest
            idx_delta = np.logical_and(freqs >= 0.0, freqs <= 4.0) 
            idx_theta = np.logical_and(freqs >= 4.0, freqs <= 8.0)
            idx_alpha = np.logical_and(freqs >= 8.0, freqs <= 13.0)
            idx_beta = np.logical_and(freqs >= 13.0, freqs <= 30.0)
            idx_low_gamma = np.logical_and(freqs >= 30.0, freqs <= 50.0)
            idx_mid_gamma = np.logical_and(freqs >= 50.0, freqs <= 70.0)
            idx_hi_gamma = np.logical_and(freqs >= 70.0, freqs <= 90.0)
            # Absolute/relative power
            f_window['f_delta_power'] = simpson(psd[idx_delta], dx = freqs_res)
            f_window['f_delta_rel_power'] = f_window['f_delta_power'] / f_window['f_total_power']
            f_window['f_theta_power'] = simpson(psd[idx_theta], dx = freqs_res)
            f_window['f_theta_rel_power'] = f_window['f_theta_power'] / f_window['f_total_power']
            f_window['f_alpha_power'] = simpson(psd[idx_alpha], dx = freqs_res)
            f_window['f_alpha_rel_power'] = f_window['f_alpha_power'] / f_window['f_total_power']
            f_window['f_beta_power'] = simpson(psd[idx_theta], dx = freqs_res)
            f_window['f_beta_rel_power'] = f_window['f_beta_power'] / f_window['f_total_power']
            f_window['f_low_gamma_power'] = simpson(psd[idx_low_gamma], dx = freqs_res)
            f_window['f_low_gamma_rel_power'] = f_window['f_low_gamma_power'] / f_window['f_total_power']
            f_window['f_mid_gamma_power'] = simpson(psd[idx_mid_gamma], dx = freqs_res)
            f_window['f_mid_gamma_rel_power'] = f_window['f_mid_gamma_power'] / f_window['f_total_power']
            f_window['f_hi_gamma_power'] = simpson(psd[idx_hi_gamma], dx = freqs_res)
            f_window['f_hi_gamma_rel_power'] = f_window['f_hi_gamma_power'] / f_window['f_total_power']

            # pwr -> power ratio
            f_window['f_pwr_dt'] = f_window['f_delta_power'] / f_window['f_theta_power']
            f_window['f_pwr_da'] = f_window['f_delta_power'] / f_window['f_alpha_power']
            f_window['f_pwr_db'] = f_window['f_delta_power'] / f_window['f_beta_power']
            f_window['f_pwr_dlg'] = f_window['f_delta_power'] / f_window['f_low_gamma_power']
            f_window['f_pwr_dmg'] = f_window['f_delta_power'] / f_window['f_mid_gamma_power']
            f_window['f_pwr_dhg'] = f_window['f_delta_power'] / f_window['f_hi_gamma_power']
            f_window['f_pwr_td'] = f_window['f_theta_power'] / f_window['f_delta_power']
            f_window['f_pwr_ta'] = f_window['f_theta_power'] / f_window['f_alpha_power']
            f_window['f_pwr_tb'] = f_window['f_theta_power'] / f_window['f_beta_power']
            f_window['f_pwr_tlg'] = f_window['f_theta_power'] / f_window['f_low_gamma_power'] 
            f_window['f_pwr_tmg'] = f_window['f_theta_power'] / f_window['f_mid_gamma_power']
            f_window['f_pwr_thg'] = f_window['f_theta_power'] / f_window['f_hi_gamma_power']
            f_window['f_pwr_ad'] = f_window['f_alpha_power'] / f_window['f_delta_power']
            f_window['f_pwr_at'] = f_window['f_alpha_power'] / f_window['f_theta_power']
            f_window['f_pwr_ab'] = f_window['f_alpha_power'] / f_window['f_beta_power']
            f_window['f_pwr_alg'] = f_window['f_alpha_power'] / f_window['f_low_gamma_power']
            f_window['f_pwr_amg'] = f_window['f_alpha_power'] / f_window['f_mid_gamma_power']
            f_window['f_pwr_ahg'] = f_window['f_alpha_power'] / f_window['f_hi_gamma_power']
            f_window['f_pwr_bd'] = f_window['f_beta_power'] / f_window['f_delta_power']
            f_window['f_pwr_bt'] = f_window['f_beta_power'] / f_window['f_theta_power']
            f_window['f_pwr_ba'] = f_window['f_beta_power'] / f_window['f_alpha_power']
            f_window['f_pwr_blg'] = f_window['f_beta_power'] / f_window['f_low_gamma_power']
            f_window['f_pwr_bmg'] = f_window['f_beta_power'] / f_window['f_mid_gamma_power']
            f_window['f_pwr_bhg'] = f_window['f_beta_power'] / f_window['f_hi_gamma_power']
            f_window['f_pwr_lgd'] = f_window['f_low_gamma_power'] / f_window['f_delta_power']
            f_window['f_pwr_lgt'] = f_window['f_low_gamma_power'] / f_window['f_theta_power']
            f_window['f_pwr_lga'] = f_window['f_low_gamma_power'] / f_window['f_alpha_power']
            f_window['f_pwr_lgb'] = f_window['f_low_gamma_power'] / f_window['f_beta_power']
            f_window['f_pwr_lgmg'] = f_window['f_low_gamma_power'] / f_window['f_mid_gamma_power']
            f_window['f_pwr_lghg'] = f_window['f_low_gamma_power'] / f_window['f_hi_gamma_power']
            f_window['f_pwr_mgd'] = f_window['f_mid_gamma_power'] / f_window['f_delta_power']
            f_window['f_pwr_mgt'] = f_window['f_mid_gamma_power'] / f_window['f_theta_power']
            f_window['f_pwr_mga'] = f_window['f_mid_gamma_power'] / f_window['f_alpha_power']
            f_window['f_pwr_mgb'] = f_window['f_mid_gamma_power'] / f_window['f_beta_power']
            f_window['f_pwr_mglg'] = f_window['f_mid_gamma_power'] / f_window['f_low_gamma_power']
            f_window['f_pwr_mghg'] = f_window['f_mid_gamma_power'] / f_window['f_hi_gamma_power']
            f_window['f_pwr_hgd'] = f_window['f_hi_gamma_power'] / f_window['f_delta_power']
            f_window['f_pwr_hgt'] = f_window['f_hi_gamma_power'] / f_window['f_theta_power']
            f_window['f_pwr_hga'] = f_window['f_hi_gamma_power'] / f_window['f_alpha_power']
            f_window['f_pwr_hgb'] = f_window['f_hi_gamma_power'] / f_window['f_beta_power']
            f_window['f_pwr_hglg'] = f_window['f_hi_gamma_power'] / f_window['f_low_gamma_power']
            f_window['f_pwr_hgmg'] = f_window['f_hi_gamma_power'] / f_window['f_mid_gamma_power']
            # Mean frequency
            f_window['f_meanfreq'] = meanfreq(window.to_numpy(copy = True), SAMPLE_RATE)
            # Brain indicies
            f_window['f_brain_index1'] = f_window['f_beta_power'] / (f_window['f_theta_power'] + f_window['f_alpha_power'])
            f_window['f_brain_index2'] = f_window['f_theta_power'] / (f_window['f_beta_power'] + f_window['f_alpha_power'])
            # Finding what frequency corresponds to maximum alpha frequency
            f_window['f_peak_alpha'] = np.argmax(psd[idx_alpha])

            df_chan_uf = pd.concat([df_chan_uf, pd.DataFrame(f_window, index = [0])], ignore_index = True)

        df_chan_uf = df_chan_uf.add_prefix(chan_name.lower() + '_', axis = 1)
        df_uf = pd.concat([df_uf, df_chan_uf], axis = 1)

    ## Multi-channel features
    for _, row in df_uf.iterrows():
        f_pair = {}
        for pair in ELECTRODE_PAIRS:
            x, y = pair
            # Differential asymmetry calculations
            f_pair[x + '_' +  y + '_f_da_d'] = row[x + '_f_delta_power'] - row[y + '_f_delta_power']
            f_pair[x + '_' +  y + '_f_da_t'] = row[x + '_f_theta_power'] - row[y + '_f_theta_power']
            f_pair[x + '_' +  y + '_f_da_a'] = row[x + '_f_alpha_power'] - row[y + '_f_alpha_power']
            f_pair[x + '_' +  y + '_f_da_b'] = row[x + '_f_beta_power'] - row[y + '_f_beta_power']
            f_pair[x + '_' +  y + '_f_da_lg'] = row[x + '_f_low_gamma_power'] - row[y + '_f_low_gamma_power']
            f_pair[x + '_' +  y + '_f_da_mg'] = row[x + '_f_mid_gamma_power'] - row[y + '_f_mid_gamma_power']
            f_pair[x + '_' +  y + '_f_da_hg'] = row[x + '_f_hi_gamma_power'] - row[y + '_f_hi_gamma_power']
            # Absolute asymmetry calculations
            f_pair[x + '_' +  y + '_f_aa_d'] = row[x + '_f_delta_power'] / row[y + '_f_delta_power']
            f_pair[x + '_' +  y + '_f_aa_t'] = row[x + '_f_theta_power'] / row[y + '_f_theta_power']
            f_pair[x + '_' +  y + '_f_aa_a'] = row[x + '_f_alpha_power'] / row[y + '_f_alpha_power']
            f_pair[x + '_' +  y + '_f_aa_b'] = row[x + '_f_beta_power'] / row[y + '_f_beta_power']
            f_pair[x + '_' +  y + '_f_aa_lg'] = row[x + '_f_low_gamma_power'] / row[y + '_f_low_gamma_power']
            f_pair[x + '_' +  y + '_f_aa_mg'] = row[x + '_f_mid_gamma_power'] / row[y + '_f_mid_gamma_power']
            f_pair[x + '_' +  y + '_f_aa_hg'] = row[x + '_f_hi_gamma_power'] / row[y + '_f_hi_gamma_power']

        f_pair['f_brainbeat'] = row['fz_f_theta_power'] / row['pz_f_alpha_power']
        df_mf = pd.concat([df_mf, pd.DataFrame(f_pair, index = [0])], ignore_index = True)

    df = pd.concat([df_uf, df_mf], axis = 1)
    return df