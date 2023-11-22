import os
from datetime import datetime

import gin

import numpy as np

from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, make_scorer
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import torch
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from models.cnn import CNNClassifier
from models.rnn import RNNClassifier

from utils import perclass_accuracy

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
            xgb.XGBClassifier(
                eval_metric = wandb.config['xgb_eval_metric'],
                objective = wandb.config['xgb_objective'],
                max_depth = wandb.config['xgb_max_depth'],
                tree_method = wandb.config['xgb_tree_method'],
                random_state = rng
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