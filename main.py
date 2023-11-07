import os
import sys
from datetime import datetime


import argparse
import gin
import torch
import numpy as np
from pytorch_lightning.trainer import seed_everything

from dataloader import MedeirosDataset, MedeirosDatasetRaw, PeitekDataset, PeitekDatasetRaw
import train

@gin.configurable('run')
def main(
    fixed_seed = gin.REQUIRED,
    cv_folds = gin.REQUIRED,
    cv_repetitions = gin.REQUIRED,
    epochs = gin.REQUIRED,
    dataset_name = gin.REQUIRED,
    model_name = gin.REQUIRED
):
    logger = print
    # Take current time for logging
    start_time = datetime.now()
    logger(f'Started program at {start_time}')
    # Define and create paths to store run data
    logs_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'logs'
    )
    try:
        os.makedirs(logs_dir, exist_ok = True)
    except OSError as e:
        sys.exit(f'Error creating logging directory: {e}')

    checkpoints_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'checkpoints'
    )
    try:
        os.makedirs(checkpoints_dir, exist_ok = True)
    except OSError as e:
        sys.exit(f'Error creating checkpoint directory: {e}')

    # Set fixed seed if needed
    if fixed_seed is not None:
        torch.manual_seed(fixed_seed)
        np.random.seed(fixed_seed)
        seed_everything(fixed_seed, workers = True)
        logger(f'Set fixed seed {fixed_seed}')

    if dataset_name == 'medeiros':
        dataset = MedeirosDataset()
    elif dataset_name == 'medeiros_raw':
        dataset = MedeirosDatasetRaw()
    elif dataset_name == 'peitek':
        dataset = PeitekDataset()
    elif dataset_name == 'peitek_raw':
        dataset = PeitekDatasetRaw()
    logger(f'Using dataset {dataset_name}')

    # Start the model fit or training depending on requested model
    if model_name in ['svm', 'xgboost']:
        logger(f'Starting fit: {model_name} with {cv_folds} folds / {cv_repetitions} repetitions')
        train.fit(dataset, model_name, cv_folds, cv_repetitions, logger, checkpoints_dir)
    else:
        logger(f'Starting training for {epochs} epochs: {model_name} with {cv_folds} folds / {cv_repetitions} repetitions')
        train.train_test_loop(dataset, model_name, epochs, cv_folds, cv_repetitions, logger, checkpoints_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog = 'Reproduction program for Master\'s thesis',
        description = '''
        This is the repreduction program for the Master\'s thesis work completed by Alisher Turubayev (supervisor - Fabian Stolp) 
        as part of the Digital Health program at Hasso-Plattner Institute. The program includes code to reproduce experiments on
        4 models - Support Vector Machines (SVM), gradient-boosting trees XGBoost, Recurrent and Convolutional Neural Networks (RNN, CNN).
        In addition, the folder `scripts` contains Python and Matlab scripts to preprocess data from two sources: Medeiros et al. (2021) 
        and Peitek et al. (2022).
        '''
    )

    parser.add_argument(
        '-m',
        '--model',
        choices = ['svm', 'xgboost', 'rnn', 'cnn'],
        required = True,
        help = 'Model to fit/train'
    )
    parser.add_argument(
        '-d',
        '--dataset',
        choices = ['medeiros', 'medeiros_raw', 'peitek', 'peitek_raw'],
        required = True,
        help = 'Dataset to use for experiment. `raw` suffix denotes datasets with raw data for DL models, such as CNN/RNN'
    )

    args = vars(parser.parse_args())
    model_name = args['model']
    dataset_name = args['dataset']

    assert not ((model_name == 'svm' or model_name == 'xgboost') and ('raw' in dataset_name)), 'Cannot use `raw` dataset with ML models'

    try:
        gin.parse_config_file(f'configs/models/{model_name}.gin')   
        gin.parse_config_file(f'configs/datasets/{dataset_name}.gin') 
    except:
        sys.exit('Error parsing the configuration file. Make sure you are not missing Gin configuration files in `configs` directory')
    main()