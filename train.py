import os
from datetime import datetime

import gin

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC

from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from models.cnn import CNNClassifier
from models.rnn import RNNClassifier

import xgboost as xgb

@gin.configurable('train_params')
def train_test_loop(dataset, model_name, epochs, cv_folds, cv_repetitions, logger, checkpoint_path):
    # Determine the shape of the data to initalize our DL model
    data_shape, label_shape = dataset.get_data_shape()
    # Input size is # of data channels (if the dataset is in raw format) or # of features
    input_size = data_shape[0]
    # Output size is # of classes to predict
    output_size = label_shape[0]
    # Window size is # of datapoints per sample (if the dataset is in raw format) or None to indicate that data is tabular
    window_datapoints = data_shape[1] if dataset.is_raw else None
    # Make a split of the dataset
    train_dataset, val_dataset, test_dataset = random_split(dataset, [0.6, 0.2, 0.2])

    # Cache to RAM both the train and validation datasets - they are accessed each epoch
    #   Because the ouput of random_split is a class `Subset`, we need to access its' property rather than calling function directly
    train_dataset.dataset.cache_to_ram()
    val_dataset.dataset.cache_to_ram()

    # Initialize dataloaders
    train_loader = DataLoader(train_dataset, batch_size = 4, shuffle = True, num_workers = min(os.cpu_count(), 4))
    val_loader = DataLoader(val_dataset, batch_size = 4, shuffle = False, num_workers = min(os.cpu_count(), 4))
    test_loader = DataLoader(test_dataset, batch_size = 4, shuffle = False, num_workers = min(os.cpu_count(), 4))

    if model_name == 'cnn':
        model = CNNClassifier(input_size, output_size, window_datapoints)
    else: 
        model = RNNClassifier(input_size, output_size, window_datapoints)

    # Enable checkpointing
    checkpoint_callback = ModelCheckpoint(
        monitor = 'val_acc',
        dirpath = checkpoint_path,
        filename = model_name + '-' + dataset.__class__.__name__ + '-{epoch:02d}-{val_acc:.2f}'
    )
    # Enable early stopping based on per-epoch validation accuracy
    earlystopping_callback = EarlyStopping(
        monitor = 'val_acc',
        mode = 'max',
        check_finite = True
    )

    wandb_logger = WandbLogger(project = "dh-eeg-thesis")
    trainer = pl.Trainer(logger = wandb_logger, max_epochs = epochs, callbacks = [checkpoint_callback, earlystopping_callback])
    
    logger('--------------------')
    logger('Starting training...')

    start_time = datetime.now()
    
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