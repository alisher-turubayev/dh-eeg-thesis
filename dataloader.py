import gin
import pandas as pd
import numpy as np
from sklearn import preprocessing

class DataLoader():
    def get_data(self):
        raise NotImplementedError

@gin.configurable('SampleDataset')
class SampleDataloader(DataLoader):
    # Predefined features list for which to calculate second-order features
    NUMERIC_FEATURES = ['Attention', 'Mediation', 'Raw', 'Delta', 'Theta', 'Alpha1', 'Alpha2', 'Beta1', 'Beta2', 'Gamma1', 'Gamma2']

    def __init__(self, path = gin.REQUIRED, n_segments = gin.REQUIRED):
        super().__init__()
        self.path = path

        # Load and preprocess data
        self.preprocess(self.load(), n_segments)

    def load(self):
        return pd.read_csv(self.path)

    def preprocess(self, df, n_segments):
        # Drop rows with na values
        df.dropna(axis = 0, inplace = True)
        
        # If number of segments is 0, use the dataset as-is
        if n_segments == 0:
            # Normalize the x values
            self.x = preprocessing.normalize(df.drop(columns = ['SubjectID', 'VideoID', 'user-definedlabeln']).values)
            self.y = df[['user-definedlabeln']].values
        
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
        
        return
        
    def get_data(self):
        return self.x, self.y

@gin.configurable('MedeirosDataset')
class MedeirosDataloader():
    def load():
        pass

@gin.configurable('MedeirosRawDataset')
class MedeirosRawDataloader():
    def load():
        pass