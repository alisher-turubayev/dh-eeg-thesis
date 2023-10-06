import os
from datetime import datetime

import gin

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer, accuracy_score, f1_score, precision_score, recall_score
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle

from shutil import rmtree
from joblib import Memory

import pickle

from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from models.cnn import CNNClassifier
from models.rnn import RNNClassifier

import xgboost as xgb

import wandb

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
        mode = 'max',
        dirpath = checkpoint_path,
        filename = model_name + '-' + dataset.__class__.__name__ + '-{epoch:02d}-{val_acc:.2f}',
        verbose = True
    )
    # Enable early stopping
    earlystopping_callback = EarlyStopping(
        patience = 10,
        monitor = 'val_step_loss',
        mode = 'min',
        check_finite = True,
        verbose = True
    )

    wandb_logger = WandbLogger(project = "dh-eeg-thesis")
    trainer = pl.Trainer(logger = wandb_logger, max_epochs = epochs, callbacks = [checkpoint_callback, earlystopping_callback])
    
    logger('--------------------')
    logger('Starting training...')

    start_time = datetime.now()
    
    trainer.fit(model, train_dataloaders = train_loader, val_dataloaders = val_loader)

    end_time = datetime.now()

    logger('-------------------')
    logger(f'Training complete. Total time to train: {end_time - start_time}')

    logger('Starting testing...')
    start_time = datetime.now()
    trainer.test(model, dataloaders = test_loader)
    end_time = datetime.now()
    logger(f'Testing complete. Time to test {end_time - start_time}')

    return

def fit(dataset, model_name, cv_folds, cv_repetitions, logger):
    # Used resources:
    # https://scikit-learn.org/stable/auto_examples/compose/plot_compare_reduction.html
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html
    # https://docs.wandb.ai/guides/integrations/scikit

   
    X, y = dataset.get_data()
    # Shuffle input array
    X, y = shuffle(X, y)
    # Make sure that the one-hot encoding is reversed before feeding the data into the ML algorithms
    y.rename(columns = {'label0': 0, 'label1': 1, 'label2': 2, 'label3': 3,}, inplace = True)
    y = y.idxmax(1)

    # None forces the n_components to be min(n_samples, n_features)
    N_COMPONENTS = [1, 5, None]
    SVM_C = [pow(2, -10), pow(2, -5), pow(2, 1), pow(2, 5), pow(2, 10)]
    SVM_MAX_ITER = [1000, 2000, 5000]

    pipe = Pipeline(
        [
            ('scaling', StandardScaler()),
            ('reduce_dim', 'passthrough'),
            ('classifier', 'passthrough'),
        ]
    )

    if model_name == 'svm':
        grid_params = [
            {
                'reduce_dim': [PCA()],
                'reduce_dim__n_components': N_COMPONENTS, 
                'classifier': [LinearSVC()],
                'classifier__dual': [True],
                'classifier__C': SVM_C,
                'classifier__max_iter': SVM_MAX_ITER
            }
        ]
        config = {
            'PCA_n_components': N_COMPONENTS,
            'classifier': 'SVM',
            'classifier_dual': 'auto',
            'classifier_C': SVM_C,
            'classifier_max_iter': SVM_MAX_ITER
        }
    else:
        grid_params = [
            {
                'reduce_dim': [PCA()],
                'reduce_dim__n_components': N_COMPONENTS, 
                'classifier': [xgb.XGBClassifier()]
                # TODO: add classifier hyperparameters
            }
        ]
        config = {
            'PCA_n_components': N_COMPONENTS,
            'classifier': 'XGBoost'
        }

    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average = 'macro', zero_division = 0),
        'recall': make_scorer(recall_score, average = 'macro', zero_division = 0),
        'f1_score': make_scorer(f1_score, average = 'macro', zero_division = 0)    
    }

    gs = GridSearchCV(
        estimator = pipe,
        param_grid = grid_params,
        scoring = scoring,
        refit = 'accuracy',
        cv = cv_repetitions,
        verbose = 1,
        n_jobs = -1
    )

    wandb.init(project = "dh-eeg-thesis", config = config)

    gs.fit(X, y)
    results = gs.cv_results_

    wandb.run.summary['best_accuracy'] = gs.best_score_
    
    wandb.log({
        'mean_fit_time': results['mean_fit_time'],
        'mean_score_time': results['mean_score_time'],
        'mean_accuracy': results['mean_test_accuracy'],
        'mean_precision': results['mean_test_precision'],
        'mean_recall': results['mean_test_recall'],
        'mean_f1_score': results['mean_test_f1_score'],
    })
    return