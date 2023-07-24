import gin
import pandas as pd
import numpy as np
from sklearn import preprocessing
import torch
from torch.utils.data import Dataset
 
@gin.configurable('SampleDataset')
class SampleDataset(Dataset):
    # TODO: move all this code to preprocessing script - it's not needed here
    # i.e. move all segments' splicing, data selection, etc. 
    # The goal is to have the user specify data path to the dataset rather than be be-all/fit-all approach here.
    
    # Predefined features list for which to calculate second-order features
    NUMERIC_FEATURES = ['Attention', 'Mediation', 'Raw', 'Delta', 'Theta', 'Alpha1', 'Alpha2', 'Beta1', 'Beta2', 'Gamma1', 'Gamma2']

    def __init__(self, device, path = gin.REQUIRED, n_segments = gin.REQUIRED):
        super().__init__()
        self.path = path
        self.is_raw = (n_segments != 0)

        # Load and preprocess data
        self._preprocess(self._load(), n_segments)

    def _load(self):
        return pd.read_csv(self.path)

    def _preprocess(self, df, n_segments):
        # Drop rows with na values
        df.dropna(axis = 0, inplace = True)
        
        # If number of segments is 0, use the dataset as-is
        if n_segments == 0:
            # Normalize the x values
            self.x = preprocessing.normalize(df.drop(columns = ['user-definedlabeln']).values)
            self.y = df[['user-definedlabeln']].values
            # As-is means we have time-series data, so the number of samples is # of participants * # of experiment conditions
            self._len = df[['SubjectID']].count() * df[['VideoID']].count()
        
        # If number of segments is 1, calculate second-order features for entire dataset
        elif n_segments == 1:
            # Group by SubjectID, VideoID and user-definedlabeln (needed to ensure that label is left within the dataset)
            # Apply aggregation functions (min, max, mean, std, median) only on numeric features
            # Reset index to remove multiindex for SubjectID, VideoID and user-definedlabeln
            df_segments = df.groupby(by = ['SubjectID', 'VideoID', 'user-definedlabeln'], as_index = False)[self.NUMERIC_FEATURES].agg(['min', 'max', 'mean', 'std', 'median']).reset_index()
            
            # Join column names to flatten the dataframe
            # https://towardsdatascience.com/how-to-flatten-multiindex-columns-and-rows-in-pandas-f5406c50e569
            df_segments.columns = ['_'.join(col) for col in df_segments.columns.values]
            
            # Remove unnecessary columns
            df_segments.drop(columns = ['SubjectID_', 'VideoID_'], inplace = True)
            
            # Split the x and y in the dataset
            self.x = preprocessing.normalize(df_segments.drop(columns = ['user-definedlabeln_']).values)
            self.y = df_segments[['user-definedlabeln_']].values
            # The number of samples is the first dimension of the x array
            self._len = self.x.shape[0]

        # Otherwise, sorcery
        else:
            # Group by subjectID, VideoID and user-definedlabeln
            df = df.groupby(by = ['SubjectID', 'VideoID', 'user-definedlabeln'], as_index = False)
            
            # Create an empty dataframe to store transformed data
            df_segments = pd.DataFrame()
            
            for group in df.groups:
                # For each group in the dataframe split the group into segments 
                segments = np.array_split(df.get_group(group), n_segments)
            
                # For each segment, calculate second-order features
                for index in range(n_segments):
                    segments[index] = segments[index][self.NUMERIC_FEATURES].agg(['min', 'max', 'mean', 'std', 'median'])
            
                    # Because our segment is now a dataframe where features are indices and columns are our original features, we need to stack it
                    #   into a Series
                    segments[index] = segments[index].stack()
            
                    # Join index names to flatten the Series
                    # https://towardsdatascience.com/how-to-flatten-multiindex-columns-and-rows-in-pandas-f5406c50e569
                    segments[index].index = ['_'.join(row) for row in segments[index].index.values]
            
                    # Add back the user-definedlabeln (it was removed when we were calculating second-order features)
                    segments[index] = pd.concat([segments[index], pd.Series([group[2]], index = ['user-definedlabeln'])])
                
                # Add newly computed segment second-order features to the dataframe 
                df_segments = pd.concat([df_segments, pd.DataFrame(segments)], ignore_index = True, copy = False)

            # Split the x and y in the dataset
            self.x = preprocessing.normalize(df_segments.drop(columns = ['user-definedlabeln']).values)
            self.y = df_segments[['user-definedlabeln']].values
            # The number of samples is the first dimension of the x array
            self._len = self.x.shape[0]
        
        return

    """
    Returns the number of samples in the dataset
    """
    def __len__(self):
        return self._len

    """
    Returns a single item from the dataset at a specified index

    TODO: validate if works:
            1. rotate x tensor? for CNN at least
    """
    def __getitem__(self, idx):
        if self.is_raw:
            return torch.tensor(self.x.iloc[idx].values), torch.tensor(self.y.iloc[idx].values)
        else:
            # Steps:
            # 1. Take a single test condition (single participant, single video)
            # 2. Delete SubjectID & VideoID columns - they are not needed
            # 3. Return it as tensor (no windowing as of now cause the dataset is too small)
            
            # Calculate the requested Subject ID and Video ID 
            subj_id = idx / self.x[['VideoID']].count()
            vid_id = idx / subj_id

            # Reindex to 0-index
            subj_id -= 1
            vid_id -= 1
            
            # Create a resulting tensor and return it
            result_df = self.x[self.x['SubjectID'] == subj_id & self.x['VideoID'] == vid_id]
            label = self.y[self.y['SubjectID'] == subj_id & self.y['VideoID'] == vid_id].iloc[0]
            return torch.tensor(result_df), torch.tensor(label)

    """ 
    Returns the entire preprocessed dataset. Used for ML methods
    """
    def get_data(self):
        if self.is_raw:
            return self.x, self.y
        else:
            raise NotImplementedError()
    
    """
    Returns the shape of the data to help initialize the DL methods
    """
    def get_shape(self):
        # Return a tuple of values (# of columns and # of classes)
        return self.x.shape[1], self.y.shape[1]

@gin.configurable('MedeirosDataset')
class MedeirosDataset(Dataset):
    """
    Initalize Medeiros dataset

    Parameters:
    path: path to dataset (must be .parquet file format)
    is_raw: whether the .parquet file contains raw (i.e. time-series signal data) or processed (i.e. tabular) data
    window_size: in seconds, the size of the window (ignored if is_raw is False)
    overlap_size: no-op TODO in float (i.e. 0.5 for 50% overlap) the size of the window overlap
    sample_rate: the sample rate of time-series signal data (ignored if is_raw is False)
    """
    def __init__(
            self,
            device,
            path = gin.REQUIRED,
            is_raw = gin.REQUIRED,
            window_size = gin.REQUIRED,
            overlap_size = gin.REQUIRED,
            sample_rate = gin.REQUIRED
            ):
        super().__init__()
        # Store all parameters 
        self.is_raw = is_raw
        self.window_size = window_size if is_raw else None
        self.overlap_size = overlap_size
        self.sample_rate = sample_rate if is_raw else None
        self.device = device

        self._preprocess(self._load(path))

    def _load(self, path):
        return pd.read_parquet(path, engine = 'fastparquet')

    def _preprocess(self, df: pd.DataFrame):
        if self.is_raw:
            # Extract all label columns (one-hot encoded)
            label_columns = [columns for columns in df.columns if columns.startswith('label')]

            self.x = df.drop(label_columns, axis = 1)
            self.y = df[label_columns]

            if self.overlap_size == .0:
                self.n_samples = int(self.x.shape[0] / (self.window_size * self.sample_rate))
            else:
                # TODO: no overlap calculations right now, add those
                raise NotImplementedError()
        else:
            raise NotImplementedError()

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        if self.is_raw:
            # TODO: no overlap calculations right now, add those
            # Calculate window boundaries
            start_time = idx * self.window_size * self.sample_rate
            end_time = start_time + self.window_size * self.sample_rate

            data = torch.tensor(self.x.iloc[start_time:end_time, :].values)
            label = torch.tensor(self.y.iloc[start_time, :].values)

            # Move to device
            data = data.to(self.device)
            label = label.to(self.device)

            # Cast both to tensor double type
            # Weird behaviour - not sure why .double() doesn't work but .float does
            data = data.float()
            label = label.float()
            # Transpose from [n_rows, n_cols] to [n_cols, n_rows] as second dim is # of channels
            return torch.transpose(data, 0, 1), label
        else:
            raise NotImplementedError()

    def get_data(self):
        if self.is_raw:
            raise TypeError('Dataset is not in tabular form')
        else:
            return self.x, self.y

    def get_shape(self):
        # Return a tuple of values (# of columns and # of classes)
        return self.x.shape[1], self.y.shape[1]

@gin.configurable('PeitekDataset')
class PeitekDataset(Dataset):
    def __len__(self):
        raise NotImplementedError()
    
    def __getitem__(self, idx):
        raise NotImplementedError()

    def get_data(self):
        raise NotImplementedError()