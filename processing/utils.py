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
import threading
import pandas as pd

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

class LockableDataFrame:
    """
    Thread-safe dataframe wrapper class.
    """
    def __init__(self):
        self._df = pd.DataFrame()
        self._lock = threading.Lock()

    def write(self, data: pd.DataFrame) -> bool:
        """
        Concatenates data to the end of the internal dataframe.

        Parameters:
        data: `pandas.DataFrame` - data to concatenate to the dataframe
        """
        with self._lock:
            self._df = pd.concat([self._df, data], ignore_index = True)
        return True

    def get_state(self) -> pd.DataFrame:
        """
        Returns the current internal dataframe.
        """
        with self._lock:
            return self._df

    def dump_to_file(self, file_path: os.PathLike) -> bool:
        """
        Saves information from the internal dataframe to file.

        Parameters:
        file_path: `os.PathLike` - path to store data in.
        """
        with self._lock:
            self._df.to_parquet(file_path)

        return True
