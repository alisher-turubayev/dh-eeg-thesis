"""
Preprocessing module for EEG datasets.

Can be used separately from the ML/DL training/evaluation module.

Allows for:
1. EEG filtering (low, high, notch)
2. Event processing/class tagging
3. Automated artifact removal (with ICA)
4. Segmenting (Medeiros et al. 2021)
5. .parquet or .set storage

Refer to README for more details
"""
import argparse
import concurrent.futures as cf
from collections.abc import Callable
import functools
from datetime import datetime
import platform
import traceback
import os
import sys
from warnings import warn
import scipy
import config
import mne

from processing.utils import detect_files_to_process
import processing.functions #pylint: disable=unused-import

EXPORT_PATH = '.\\export' if platform.system() == 'Windows' else './export'
DEBUG_FLAG = True

def process_file(
        file_path: os.PathLike,
        sample_freq: int,
        unpack_func: Callable,
        preprocess_func: Callable = None) -> str:
    # pylint: disable=not-callable
    # pylint: disable=broad-exception-caught
    """
    Processes a specified EEG file. 

    Accepts .mat, .set files. 
    """
    mne.set_log_level(verbose = False, return_old_level = False)
    try:
        if os.path.splitext(file_path) == '.set':
            data = mne.io.read_raw_eeglab(file_path, preload = True)
        else:
            data = scipy.io.loadmat(file_path)
            # Define variables - unpack the originally packed data
            eeg_raw, eeg_chan_info, eeg_events_table = unpack_func(data)

            # Create info structure needed for MNE
            info = mne.create_info(
                ch_names = eeg_chan_info, sfreq = sample_freq, ch_types = 'eeg')

            # Create RawArray object
            raw = mne.io.RawArray(eeg_raw, info)

            # Create Annotations object
            annotations = mne.Annotations(
                onset = eeg_events_table['latency'],
                duration = 0,
                description = eeg_events_table['type'])
            raw.set_annotations(annotations)

        if preprocess_func is not None:
            preprocess_func(raw)

        # Save the processed dataset under the same name (for consistency)
        savefile_name, _ = os.path.splitext(os.path.split(file_path)[1])
        mne.export.export_raw(
            os.path.join(EXPORT_PATH, savefile_name + '.set'),
            raw, fmt = 'eeglab', overwrite = True)
        return 'ok'

    except TypeError:
        print(f'(!) TypeError encountered processing file {file_path}. Aborting...')
        with open('error.log', 'a', encoding = 'utf-8') as log_file:
            log_file.write(f'TypeError on file {file_path} at {datetime.now()}\n')
            log_file.write(traceback.format_exc())
            log_file.write('\n')
        return 'error'

    except Exception as e:
        print(f'(!) Exception {type(e)} encountered processing file {file_path}. Aborting...')
        with open('error.log', 'a', encoding = 'utf-8') as log_file:
            log_file.write(f'Exception {type(e)} on file {file_path} at {datetime.now()}\n')
            log_file.write(traceback.format_exc())
            log_file.write('\n')
        return 'error'

def start_pool(cfg_dic: config.Config):
    """
    Starts a `concurrent.futures.ProcessPoolExecutor` to process files.
    """
    files = detect_files_to_process(cfg_dic['dir_path'], file_format = cfg_dic['file_format'])
    with cf.ProcessPoolExecutor(max_workers = 2) as executor:
        worker = functools.partial(
            process_file,
            sample_freq = cfg_dic['sample_freq'],
            unpack_func = cfg_dic['unpack_func'],
            preprocess_func = cfg_dic['preprocess_func'])
        for file, status in zip(files, executor.map(worker, files)):
            print(f'File {file} processing status: {status}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = 'EEG file preprocessor')
    parser.add_argument('-c', '--config', required = True,
                        help = '''Name of the gin configuration file.
                        Refer to configs/README.md for creating custom configuration files.''')

    args, _ = parser.parse_known_args()

    # Config parser
    try:
        cfg = config.Config(os.path.join(
            os.getcwd(), 'configs', 'processing', f'{args.config}.cfg'), encoding = 'utf-8')
    except IOError:
        sys.exit('Error parsing configuration file.')

    try:
        os.makedirs(EXPORT_PATH)
    except OSError:
        warn('Export path already exists.')
        input('Press Enter to continue (files will be OVERRIDEN!)')

    start_pool(cfg)
