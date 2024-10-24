"""
Data preparation script for EEG datasets.

Can be used separately from the ML/DL training/evaluation module.

Refer to README for more details
"""
import argparse
import concurrent.futures as cf
from datetime import datetime
import functools
import importlib
import traceback
import os
import sys
from warnings import warn
import config
import mne
import scipy

from processing.utils import detect_files_to_process

def process_file(
        file_path: os.PathLike,
        export_path: os.PathLike,
        unpack_func: callable,
        preprocess_func: callable = None) -> str:
    # pylint: disable=broad-exception-caught
    """
    Processes a specified EEG file. 

    Accepts .mat, .set files. 
    """
    mne.set_log_level(verbose = False, return_old_level = False)
    try:
        if os.path.splitext(file_path) == '.set':
            raw = mne.io.read_raw_eeglab(file_path, preload = True)
        else:
            data = scipy.io.loadmat(file_path)
            # Call unpack function to load and unpack the data
            raw = unpack_func(data)

        if preprocess_func is not None:
            preprocess_func(raw)

        # Save the processed dataset under the same name (for consistency)
        savefile_name, _ = os.path.splitext(os.path.split(file_path)[1])
        mne.export.export_raw(
            os.path.join(export_path, savefile_name + '.set'),
            raw, fmt = 'eeglab', overwrite = True
        )

        return 'ok'

    except Exception as e:
        print(f'(!) Exception {type(e)} encountered processing file {file_path}')
        with open('error.log', 'a', encoding = 'utf-8') as log_file:
            log_file.write(f'Exception {type(e)} on file {file_path} at {datetime.now()}\n')
            log_file.write(traceback.format_exc())
            log_file.write('\n')
        return 'error'

def start_pool(cfg_dic: config.Config, num_workers: int = 2):
    """
    Starts a `concurrent.futures.ProcessPoolExecutor` to process files.
    """
    files = detect_files_to_process(cfg_dic['data_path'], file_format = cfg_dic['file_format'])
    print(f'--- Files detected: {len(files)}.')
    # Import package for processing
    importlib.import_module(cfg_dic['module_name'])

    files_processed = 0
    start_time = datetime.now()

    print('--- Started preprocessing (process may take a while) ---')

    with cf.ProcessPoolExecutor(max_workers = num_workers) as executor:
        worker = functools.partial(
            process_file,
            export_path = cfg_dic['export_path'],
            sample_freq = cfg_dic['sample_freq'],
            unpack_func = cfg_dic['unpack_func'],
            preprocess_func = cfg_dic['preprocess_func'])
        for file, status in zip(files, executor.map(worker, files)):
            print(f'File {file} processing status: {status}')
            if status == 'ok':
                files_processed += 1

    print(f"""--- Preprocessing complete.
          Time taken: {datetime.now() - start_time}. 
          Files processed successfully {files_processed} ---""")

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
        os.makedirs(cfg['export_path'])
    except OSError:
        warn('Export path already exists.')
        input('Press Enter to continue (files will be OVERRIDEN!)')

    start_pool(cfg)
