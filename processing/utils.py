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
import glob
import os

def detect_files_to_process(search_dir: os.PathLike, file_format = '.mat'):
    """
    Returns a list of files in the directory that match the file format specified.
    Note that the search is recursive - search in a large directory tree may take inordinate amount 
    of time.

    If file_format is not specified, defaults to *.mat files.
    """
    if not os.path.exists(search_dir):
        raise IOError(f'{search_dir} not found')

    pattern = os.path.join('**', '*' + file_format)

    return [os.path.join(search_dir, found_file) for found_file
            in glob.glob(pattern, root_dir = search_dir, recursive = True)]
