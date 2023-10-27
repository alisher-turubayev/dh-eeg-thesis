import os
from datetime import datetime
import pickle
import warnings

import gin

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import make_scorer, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle

import torch
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
    # Determine if GPU acceleration is available
    if torch.cuda.is_available(): 
        dev = "cuda:0" 
    else: 
        dev = "cpu" 
    device = torch.device(dev)

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

    #wandb.finish()
    return

@gin.configurable('fit_params')
def fit(dataset, model_name, cv_folds, cv_repetitions, logger, checkpoint_path, grid_params = gin.REQUIRED, config = gin.REQUIRED):
    """
    Completes ML model fit and reports the score to the W&B dashboard.

    Reported scores:
    1. accuracy
    2. precision
    3. recall
    4. F1 score

    Used resources:
    * https://scikit-learn.org/stable/auto_examples/compose/plot_compare_reduction.html
    * https://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html
    * https://docs.wandb.ai/guides/integrations/scikit
    * https://www.kaggle.com/code/sinanhersek/why-use-repeated-cross-validation
    """
    # Suppress warnings
    warnings.filterwarnings(action = 'ignore', category = ConvergenceWarning)

    # Load dataset
    X, y = dataset.get_data()
    # Make sure that the one-hot encoding is reversed before feeding the data into the ML algorithms
    y.rename(columns = {'label0': 0, 'label1': 1, 'label2': 2, 'label3': 3,}, inplace = True)
    y = y.idxmax(1)
    
    config['dataset'] = type(dataset).__name__
    wandb.init(project = "dh-eeg-thesis", config = config)
    # To make sure that the results are not cross-contaminated between repetitions,
    # we re-init everything
    for i in range(cv_repetitions):
        start_time = datetime.now()
        # Re-initialize pipeline
        pipe = Pipeline(
            [
                ('scaling', StandardScaler()),
                ('reduce_dim', 'passthrough'),
                ('model', 'passthrough'),
            ]
        )
        # Re-initialize scoring
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score, average = 'macro', zero_division = 0),
            'recall': make_scorer(recall_score, average = 'macro', zero_division = 0),
            'f1_score': make_scorer(f1_score, average = 'macro', zero_division = 0)    
        }
        # Shuffle input array
        X, y = shuffle(X, y)

        # Reinitialize PCA/model
        grid_params['reduce_dim'] = [PCA()]
        if model_name == 'svm':
            grid_params['model'] = [OneVsOneClassifier(LinearSVC())]
        else:
            grid_params['model'] = [xgb.XGBClassifier()]

        # Re-init GridSearchCV
        gs = GridSearchCV(
            estimator = pipe,
            param_grid = grid_params,
            scoring = scoring,
            refit = 'accuracy',
            cv = cv_folds,
            verbose = 1,
            n_jobs = -1
        )
        # Make a model fit
        gs.fit(X, y)
        # Dump best model for fold to checkpoint directory
        fold_clf = gs.best_estimator_
        pickle.dump(fold_clf, open(os.path.join(checkpoint_path, model_name + f'rep{i}.sav'), 'wb'))
        # Get fold results
        fold_best_accuracy = gs.best_score_
        fold_results = gs.cv_results_
        # Plot CEV for PCA and log the resulting plot
        cev = np.cumsum(fold_clf['reduce_dim'].explained_variance_ratio_)
        fig = plt.figure()
        plt.style.use('seaborn-v0_8-pastel')
        plt.plot(cev)
        plt.ylabel('cumulative explained variance')
        plt.xlabel('component #')
        wandb.log({f'best_clf_pca_cev_rep{i}': wandb.Image(fig)})
        plt.close(fig)
        # Log plots for each metric
        fig = plt.figure()
        plt.style.use('seaborn-v0_8-pastel')
        plt.boxplot(fold_results['mean_test_accuracy'], labels = [f'Mean test accuracy for repetition {i}'])
        wandb.log({f'test_accuracy_rep{i}': wandb.Image(fig)})
        plt.close(fig)
        fig = plt.figure()
        plt.style.use('seaborn-v0_8-pastel')
        plt.boxplot(fold_results['mean_test_precision'], labels = [f'Mean test precision for repetition {i}'])       
        wandb.log({f'test_precision_rep{i}': wandb.Image(fig)})
        plt.close(fig)
        fig = plt.figure()
        plt.style.use('seaborn-v0_8-pastel')
        plt.boxplot(fold_results['mean_test_recall'], labels = [f'Mean test recall for repetition {i}'])
        wandb.log({f'test_recall_rep{i}': wandb.Image(fig)})
        plt.close(fig)
        fig = plt.figure()
        plt.style.use('seaborn-v0_8-pastel')
        plt.boxplot(fold_results['mean_test_f1_score'], labels = [f'Mean test F1 score for repetition {i}'])
        wandb.log({f'test_f1_score_rep{i}': wandb.Image(fig)})
        plt.close(fig)
    
        end_time = datetime.now()
        logger('-------------------')
        logger(f'Finished repetition {i}. Best accuracy: {fold_best_accuracy}. Time to run: {end_time - start_time}')
    
    wandb.finish()
    return