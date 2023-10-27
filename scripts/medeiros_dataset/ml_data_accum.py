import os
import pandas as pd

PATH = './data/medeiros/medeiros_processed'

files = [os.path.join(PATH, file) for file in os.listdir(PATH) if os.path.isfile(os.path.join(PATH, file))]
        
data = pd.DataFrame()
labels = pd.DataFrame()

df = pd.DataFrame()

for file in files:
    df = pd.concat([df, pd.read_parquet(file)], ignore_index = True)
    #os.remove(os.path.join(PATH, file))

df.to_parquet(os.path.join(PATH, 'medeiros_dataset.parquet'))
