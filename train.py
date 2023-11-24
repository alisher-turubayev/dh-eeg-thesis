import os
from datetime import datetime

import gin

import numpy as np

from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, make_scorer
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_validate
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import torch
from torch.utils.data import SubsetRandomSampler, DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from models.cnn import CNNClassifier
from models.rnn import RNNClassifier

from utils import perclass_accuracy

import xgboost as xgb

import wandb

@gin.configurable('train_params')
def train_test_loop(dataset, model_name, epochs, rng, cv_folds, cv_repetitions, logger, args):
    # Determine if GPU acceleration is available
    if torch.cuda.is_available(): 
        dev = "cuda:0" 
    else: 
        dev = "cpu" 
    device = torch.device(dev)
    
    wandb.init(project = "dh-eeg-thesis", config = args)
    # Store the dataset in memory
    dataset.cache_to_ram()

    # Determine the shape of the data to initalize our DL model
    data_shape, label_shape = dataset.get_data_shape()
    # Input size is # of data channels
    input_size = data_shape[0]
    # Output size is # of classes to predict
    output_size = label_shape[0]
    # Window size is # of datapoints per sample
    window_datapoints = data_shape[1]
    # Get non-one-hot encoded labels for StratifiedKFold
    labels_strat = dataset.get_strat_labels()
    # Make a split of the dataset
    train_val_idx, test_idx = train_test_split(range(len(dataset)), random_state = rng, test_size = 0.2, stratify = labels_strat)

    # Initialize dataloader for the test indicies
    test_loader = DataLoader(dataset, batch_size = len(test_idx), sampler = SubsetRandomSampler(test_idx), num_workers = min(os.cpu_count(), 4))

    train_val_labels = np.take(labels_strat, train_val_idx)
    # Initialize RepeatedStatifiedKFold
    skf = RepeatedStratifiedKFold(n_splits = cv_folds, n_repeats = cv_repetitions, random_state = rng)

    # Enable early stopping
    earlystopping_callback = EarlyStopping(
        patience = 10,
        monitor = 'val_step_loss',
        mode = 'min',
        check_finite = True,
        verbose = True
    )

    # Start the cross-validation
    for i, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(train_val_labels)), train_val_labels)):
        # Initialize train/val loaders
        train_loader = DataLoader(dataset, batch_size = 4, sampler = SubsetRandomSampler(train_idx), num_workers = min(os.cpu_count(), 4))
        val_loader = DataLoader(dataset, batch_size = 4, sampler = SubsetRandomSampler(val_idx), num_workers = min(os.cpu_count(), 4))

        # Initialize the model
        if model_name == 'cnn':
            model = CNNClassifier(input_size, output_size, window_datapoints)
        else: 
            model = RNNClassifier(input_size, output_size, window_datapoints, args['rnn_hidden_size'], args['rnn_n_layers'], i)

        logger('--------------------')
        logger(f'Starting fold {i}...')
        start_time = datetime.now()
        trainer = pl.Trainer(max_epochs = epochs, callbacks = [earlystopping_callback])
        trainer.fit(model, train_dataloaders = train_loader, val_dataloaders = val_loader)
        end_time = datetime.now()
        logger('-------------------')
        logger(f'Training for fold {i} complete. Total time to train: {end_time - start_time}')

        logger(f'Starting testing on fold {i}...')
        start_time = datetime.now()
        trainer.test(model, dataloaders = test_loader)
        end_time = datetime.now()
        logger(f'Testing complete for fold {i}. Time to test {end_time - start_time}')

    wandb.finish()
    return

@gin.configurable('fit_params')
def fit(dataset, model_name, rng, cv_folds, cv_repetitions, logger, args):
    """
    Completes ML model fit and reports the score to the W&B dashboard.

    Reported scores:
    1. accuracy
    2. per-class accuracy
    3. precision
    4. recall
    5. F1 score

    Used resources:
    * https://scikit-learn.org/stable/auto_examples/compose/plot_compare_reduction.html
    * https://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html
    * https://docs.wandb.ai/guides/integrations/scikit
    * https://www.kaggle.com/code/sinanhersek/why-use-repeated-cross-validation
    """
    # Load dataset
    X, y = dataset.get_data()
    wandb.init(project = "dh-eeg-thesis", config = args)
    # Initialize model
    pipeline_params = [
        ('scaling', StandardScaler()),
        ('reduce_dim', PCA(n_components = wandb.config['pca_cev'], random_state = rng)),
    ]
    if model_name == 'svm':
        pipeline_params.append((
            'model', 
            OneVsOneClassifier(
                LinearSVC(
                    dual = wandb.config['svm_dual'],
                    C = wandb.config['svm_C'],
                    max_iter = wandb.config['svm_max_iter'],
                    random_state = rng
                )
            )
        ))
    else:
        pipeline_params.append((
            'model', 
            OneVsOneClassifier(
                xgb.XGBClassifier(
                    eval_metric = wandb.config['xgb_eval_metric'],
                    objective = wandb.config['xgb_objective'],
                    max_depth = wandb.config['xgb_max_depth'],
                    tree_method = wandb.config['xgb_tree_method'],
                    random_state = rng
                )
            )
        ))
    pipe = Pipeline(pipeline_params)
    # Define metrics of interest
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'accuracy_class0': make_scorer(perclass_accuracy, class_pos = 0),
        'accuracy_class1': make_scorer(perclass_accuracy, class_pos = 1),
        'accuracy_class2': make_scorer(perclass_accuracy, class_pos = 2),
        'accuracy_class3': make_scorer(perclass_accuracy, class_pos = 3),
        'precision': make_scorer(precision_score, average = 'micro'),
        'recall': make_scorer(recall_score, average = 'micro'),
        'f1_score': make_scorer(f1_score, average = 'micro')    
    }
    # Define k-fold cross-validation and conduct it
    skf = RepeatedStratifiedKFold(n_splits = cv_folds, n_repeats = cv_repetitions, random_state = rng)
    cv_result = cross_validate(pipe, X, y, scoring = scoring, cv = skf, n_jobs = -1)
    # Log metrics of interest
    for i in range(cv_repetitions * cv_folds):
        wandb.log({
            'fold_acc': cv_result['test_accuracy'][i],
            'fold_acc_class0': cv_result['test_accuracy_class0'][i],
            'fold_acc_class1': cv_result['test_accuracy_class1'][i],
            'fold_acc_class2': cv_result['test_accuracy_class2'][i],
            'fold_acc_class3': cv_result['test_accuracy_class3'][i],
            'fold_precision': cv_result['test_precision'][i],
            'fold_recall': cv_result['test_recall'][i],
            'fold_f1_score': cv_result['test_f1_score'][i]
        })
    mean_accuracy = np.mean(cv_result['test_accuracy'])
    # Log mean test accuracy separately - this is used by W&B for hyperparameter tuning
    wandb.log({'mean_fold_acc': mean_accuracy})
    logger(f'Mean accuracy across folds/repetitions for run {wandb.run.name}/model {model_name}: {mean_accuracy}')

    # Close the W&B connection
    wandb.finish()
    return