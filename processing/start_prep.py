"""
Preprocessing script for EEG datasets.

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

from processing.utils import detect_files_to_process, LockableDataFrame

def process_participant(
        data_df: LockableDataFrame,
        available_files: list[os.PathLike],
        pack_func: callable,
        participant_id: str
    ) -> str:
    # pylint: disable=not-callable
    # pylint: disable=broad-exception-caught
    """
    Processes a specified EEG file. 

    Accepts .set files
    """
    mne.set_log_level(verbose = False, return_old_level = False)
    try:
        files = [file for file in available_files if participant_id in str(file)]
        packed_data = pack_func(files)
        data_df.write(packed_data)
        return 'ok'

    except Exception as e:
        print(f'(!) Exception {type(e)} encountered processing participant {participant_id}')
        with open('error.log', 'a', encoding = 'utf-8') as log_file:
            log_file.write(f'Exception {type(e)} on file {participant_id} at {datetime.now()}\n')
            log_file.write(traceback.format_exc())
            log_file.write('\n')
        return 'error'

def start_pool(cfg_dic: config.Config, num_workers: int = 2):
    """
    Starts a `concurrent.futures.ProcessPoolExecutor` to process files.
    """
    participants = cfg_dic['participant_ids']
    participant_processed = 0
    start_time = datetime.now()
    # Thread-locked data frame for final data
    lockable_df = LockableDataFrame()

    # Take all available files in the directory; we are doing per-subject data prep
    available_files = detect_files_to_process(cfg_dic['data_path'], file_format = '.set')

    # Import package for processing
    importlib.import_module(cfg_dic['module_name'])

    print('--- Started data preparation for ML/DL (process may take a while) ---')

    with cf.ProcessPoolExecutor(max_workers = num_workers) as executor:
        worker = functools.partial(
            process_participant,
            data_df = lockable_df,
            available_files = available_files,
            pack_func = cfg_dic['pack_func']
        )
        for participant_id, status in zip(participants, executor.map(worker, participants)):
            print(f'Participant {participant_id} processing status: {status}')
            if status == 'ok':
                participant_processed += 1

    print('--- Exporting data... ---')
    lockable_df.dump_to_file(os.path.join(cfg_dic['export_path'], cfg_dic['export_file']))

    print(f"""--- Data preparation for ML/DL complete.
          Time taken: {datetime.now() - start_time}. 
          {participant_processed} paricipants processed successfully ---""")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = 'EEG data preparation for ML/DL script')
    parser.add_argument('-c', '--config', required = True,
                        help = '''Name of the gin configuration file.
                        Refer to configs/README.md for creating custom configuration files.''')

    args, _ = parser.parse_known_args()

    # Config parser
    try:
        cfg = config.Config(os.path.join(
            os.getcwd(), 'configs', 'processing', f'{args.config}_prep.cfg'), encoding = 'utf-8')
    except IOError:
        sys.exit('Error parsing configuration file.')

    try:
        os.makedirs(cfg['export_path'])
    except OSError:
        warn('Export path already exists.')
        input('Press Enter to continue (previous file will be OVERRIDEN!)')

    start_pool(cfg)
