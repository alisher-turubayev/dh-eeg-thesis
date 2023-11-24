import os
import sys
from datetime import datetime


import argparse
import gin
import numpy as np
from pytorch_lightning.trainer import seed_everything

from dataloader import MedeirosDataset, MedeirosDatasetRaw, CustomDataset, CustomDatasetRaw
import train
import utils

@gin.configurable('run')
def main(
    args: dict[str, any],
    fixed_seed = gin.REQUIRED,
    cv_folds = gin.REQUIRED,
    cv_repetitions = gin.REQUIRED,
    epochs = gin.REQUIRED,
    dataset_name = gin.REQUIRED,
    model_name = gin.REQUIRED,
):
    logger = print
    # Take current time for logging
    start_time = datetime.now()
    logger(f'Started program at {start_time}')

    # Set fixed seed if needed
    rng = None
    if fixed_seed is not None:
        if model_name in ['svm', 'xgboost']:
            rng = np.random.RandomState(fixed_seed)
        else:
            rng = np.random.RandomState(fixed_seed)
            seed_everything(fixed_seed, workers = True)
        logger(f'Set fixed seed {fixed_seed}')

    if dataset_name == 'medeiros':
        dataset = MedeirosDataset()
    elif dataset_name == 'medeiros_raw':
        dataset = MedeirosDatasetRaw()
    elif dataset_name == 'custom':
        dataset = CustomDataset()
    elif dataset_name == 'custom_raw':
        dataset = CustomDatasetRaw()
    logger(f'Using dataset {dataset_name}')

    # Start the model fit or training depending on requested model
    if model_name in ['svm', 'xgboost']:
        logger(f'Starting fit: {model_name} with {cv_folds} folds / {cv_repetitions} repetitions')
        train.fit(dataset, model_name, rng, cv_folds, cv_repetitions, logger, args)
    else:
        logger(f'Starting training for {epochs} epochs: {model_name} with {cv_folds} folds / {cv_repetitions} repetitions')
        train.train_test_loop(dataset, model_name, epochs, rng, cv_folds, cv_repetitions, logger, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog = 'Reproduction program for Master\'s thesis',
        description = '''
        This is the reproduction program for the Master\'s thesis work completed by Alisher Turubayev (supervisor - Fabian Stolp) 
        as part of the Digital Health program at Hasso-Plattner Institute. The program includes code to reproduce experiments on
        4 models - Support Vector Machines (SVM), gradient-boosting trees XGBoost, Recurrent and Convolutional Neural Networks (RNN, CNN).
        In addition, the folder `scripts` contains Python and Matlab scripts to preprocess data from Medeiros et al (2021). https://doi.org/10.3390/s21072338
        '''
    )

    utils.add_args(parser)
    args = vars(parser.parse_args())
    utils.validate_args(args)

    model_name = args['model']
    dataset_name = args['dataset']

    try:
        gin.parse_config_file(f'configs/models/{model_name}.gin')   
        gin.parse_config_file(f'configs/datasets/{dataset_name}.gin') 
    except:
        sys.exit('Error parsing the configuration file. Make sure you are not missing Gin configuration files in `configs` directory')
    main(args)