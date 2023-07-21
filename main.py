import gin
import torch
import numpy as np
import wandb

from dataloader import SampleDataset, MedeirosDataset, PeitekDataset
import train

from datetime import datetime
import logging
import sys
import os

@gin.configurable('run')
def main(
    fixed_seed = gin.REQUIRED,
    cv_folds = gin.REQUIRED,
    cv_repetitions = gin.REQUIRED,
    epochs = gin.REQUIRED,
    dataset_name = gin.REQUIRED,
    model_name = gin.REQUIRED
):
    assert dataset_name in ['sample', 'medeiros', 'peitek'], f'Dataset name not recognized: {dataset_name}'
    assert model_name in ['svm', 'xgboost', 'cnn', 'rnn'], f'Model name not recognized: {model_name}'

    # Take current time for logging
    start_time = datetime.now()

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

    # Set up logging
    logging.basicConfig(filename = os.path.join(logs_dir, f'{start_time}.log'), encoding = 'utf-8', level = logging.INFO)    
    logging.info(f'Started program at {start_time}')

    # Set fixed seed if needed
    if fixed_seed is not None:
        torch.manual_seed(fixed_seed)
        np.random.seed(fixed_seed)
        logging.info(f'Set fixed seed {fixed_seed}')

    # Determine if GPU acceleration is available
    if torch.cuda.is_available(): 
        dev = "cuda:0" 
    else: 
        dev = "cpu" 
    device = torch.device(dev)

    # Read configuration files for dataset_name/model
    try:
        gin.parse_config_file(f'configs/datasets/{dataset_name}_dataset.gin')
        gin.parse_config_file(f'configs/models/{model_name}.gin')
    except IOError as e:
        logging.error(f': Error reading gin configuration files: {e}')
        sys.exit(1)

    if dataset_name == 'sample':
        dataset = SampleDataset(device)
    elif dataset_name == 'medeiros':
        dataset = MedeirosDataset(device)
    elif dataset_name == 'peitek':
        dataset = PeitekDataset(device)
    logging.info(f'Using dataset_name {dataset_name}')

    """
    TODO: remove this comment once most of the work on development is done
    wandb.init(
        project = 'dh-eeg-thesis',
        config = {
            "fixed_seed": fixed_seed,
            "cv_folds": cv_folds,
            "cv_repetitions": cv_repetitions,
            "dataset_name": dataset_name,
            "model_name": model_name
        }
    )
    """

    # Start the model fit or training depending on requested model
    if model_name in ['svm', 'xgboost']:
        logging.info(f'Starting fit: {model_name} with {cv_folds} folds / {cv_repetitions} repetitions')
        train.fit(dataset, model_name, cv_folds, cv_repetitions)
    else:
        logging.info(f'Starting training for {epochs} epochs: {model_name} with {cv_folds} folds / {cv_repetitions} repetitions')
        train.train_test_loop(device, dataset, model_name, epochs, cv_folds, cv_repetitions)

    wandb.finish()

if __name__ == '__main__':
    try:
        gin.parse_config_file(f'configs/run/{sys.argv[1]}.gin')    
    except:
        sys.exit('Error parsing the configuration file.\nMake sure to specify the run configuration in the first argument, e.g. \'debug\'')
    main()