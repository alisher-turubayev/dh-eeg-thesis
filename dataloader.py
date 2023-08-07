import os
import gin
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import minmax_scale

@gin.configurable('MedeirosDataset')
class MedeirosDataset(Dataset):
    def __init__(self, path = gin.REQUIRED):
        super().__init__()
        self.is_raw = False
    
    def _cache_to_ram(self):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()
    
    def __getitem__(self, idx):
        raise NotImplementedError()

    def get_data(self):
        raise NotImplementedError()

    def get_data_shape(self):
        raise NotImplementedError()

@gin.configurable('MedeirosDatasetRaw')
class MedeirosDatasetRaw(Dataset):
    """
    Initalize Medeiros dataset. See `scripts/medeiros_dataset/README.md` for details on preprocessing

    Args:
    metadata_path: pandas.DataFrame with indicies of samples to load
    path: path to dataset (must be where .parquet files are stored)
    """
    def __init__(
            self,
            metadata_path = gin.REQUIRED,
            path = gin.REQUIRED
            ):
        super().__init__()
        self.metadata = pd.read_csv(metadata_path, index_col = 0)
        self.n_samples = self.metadata.shape[0]
        self.path = path
        self.is_raw = True
        self._cached_dataset = None

    def cache_to_ram(self):
        if self._cached_dataset is None:
            self._cached_dataset = [self[i] for i in range(len(self))]

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        if self._cached_dataset is not None:
            return self._cached_dataset[idx]
        # Determine the sample_index to read from
        metadata_row = self.metadata.iloc[idx]
        sample_index = int(metadata_row.loc['sample_index'])
        # Drop unnecessary sample index column
        metadata_row.drop('sample_index', inplace = True)
        # Determine filepath
        filepath = os.path.join(self.path, str(sample_index) + '.parquet')
        # Store as tensor
        data = torch.tensor(pd.read_parquet(filepath).values)
        label = torch.tensor(metadata_row.values)
        # Cast both to tensor double type
        # Weird behaviour - not sure why .double() doesn't work but .float does
        data = data.float()
        label = label.float()
        # Transpose the [n_rows, n_cols] to [n_cols, n_rows]
        return torch.transpose(data, 0, 1), label

    # For ML, we are not implementing this method as we don't need it
    #   because we are not training on raw data anyway
    def get_data(self):
        raise NotImplementedError()

    def get_data_shape(self):
        # Get the first sample
        x, y = self[0]
        return x.shape, y.shape

@gin.configurable('PeitekDataset')
class PeitekDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.is_raw = False

    def _cache_to_ram(self):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()
    
    def __getitem__(self, idx):
        raise NotImplementedError()

    def get_data(self):
        raise NotImplementedError()

    def get_data_shape(self):
        raise NotImplementedError()

@gin.configurable('PeitekDatasetRaw')
class PeitekDatasetRaw(Dataset):
    def __init__(self):
        super().__init__()
        self.is_raw = True

    def _cache_to_ram(self):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()
    
    def __getitem__(self, idx):
        raise NotImplementedError()

    def get_data(self):
        raise NotImplementedError()

    def get_data_shape(self):
        raise NotImplementedError()