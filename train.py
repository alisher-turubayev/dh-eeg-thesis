import gin

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC

from torch.utils.data import DataLoader, random_split

from datetime import datetime

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from models.cnn import CNNClassifier
from models.rnn import RNNClassifier

import xgboost as xgb

@gin.configurable('train_params')
def train_test_loop(dataset, model_name, epochs, cv_folds, cv_repetitions, logger):
    # Get data properties to initialize the model
    input_size, output_size = dataset.get_shape()
    if dataset.is_raw:
        window_datapoints = dataset.window_size * dataset.sample_rate
    else:
        window_datapoints = None

    # Create three subsets of the original dataset
    # 60%/20%/20% split
    train_dataset, val_dataset, test_dataset = random_split(dataset, [0.6, 0.2, 0.2])
    # Initialize dataloaders
    train_loader = DataLoader(train_dataset, batch_size = 4, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = 4, shuffle = False)
    test_loader = DataLoader(test_dataset, batch_size = 4, shuffle = False)

    if model_name == 'cnn':
        model = CNNClassifier(input_size, output_size, window_datapoints)
    else: 
        model = RNNClassifier(input_size, output_size)
   
    logger('--------------------')
    logger('Starting training...')

    start_time = datetime.now()

    wandb_logger = WandbLogger(project = "dh-eeg-thesis")
    # TODO: remove enable_checkpointing = False
    trainer = pl.Trainer(logger = wandb_logger, max_epochs = epochs, enable_checkpointing = False)
    trainer.fit(model, train_dataloaders = train_loader, val_dataloaders = val_loader)

    end_time = datetime.now()

    logger('-------------------')
    logger(f'Training complete. Total time to train: {start_time - end_time}')

    trainer.test(model, dataloaders = test_loader)

    return

def fit(dataset, model_name, cv_folds, cv_repetitions, logger):
    pca = PCA()
    if model_name == 'svm':
        model = LinearSVC(C = 10)
    else:
        model = xgb.XGBClassifier()
    pipe = Pipeline(steps = [('pca', pca), ('model', model)])

    X, y = dataset.get_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)

    pipe.fit(X = X_train, y = y_train)
    logger(f'Score of {model_name}: {model.score(X = X_test, y = y_test)}')
    
    return