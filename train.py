import gin

import logging

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC

from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import torch.optim as optim
import torch

import torchmetrics.classification as metrics

from models.cnn import CNNClassifier
from models.rnn import RNNClassifier

import xgboost as xgb
import wandb

@gin.configurable('train_params')
def train_test_loop(device, dataset, model_name, epochs, cv_folds, cv_repetitions, logger, lr = 0.001, weight_decay = 0.01):
    train_dataset, val_dataset, test_dataset = random_split(dataset, [0.5, 0.3, 0.2])

    train_loader = DataLoader(train_dataset, batch_size = 4, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = 4, shuffle = False)

    if model_name == 'cnn':
        model = CNNClassifier()
    else: 
        model = RNNClassifier()
    # Move to GPU if available
    model.to(device)
    # W&B start watching the model - this will keep track of gradients
    wandb.watch(model)

    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)

    loss_fn = F.cross_entropy

    # TODO: fix hardcoded
    acc = metrics.Accuracy(task = "multiclass", num_classes = 4)
    prec = metrics.Precision(task = "multiclass", num_classes = 4)
    recall = metrics.Recall(task = "multiclass", num_classes = 4)
    f1score = metrics.F1Score(task = "multiclass", num_classes = 4)

    logger('--------------------')
    logger('Starting training...')

    for epoch in range(epochs):
        running_loss = 0.
        running_loss_val = 0.

        for idx, batch in enumerate(train_loader):
            x, y = batch

            optimizer.zero_grad()
            pred_y = model(x)

            loss = loss_fn(pred_y, y)
            
            acc.update(pred_y, y)
            prec.update(pred_y, y)
            recall.update(pred_y, y)
            f1score.update(pred_y, y)

            loss.backward()

            optimizer.step()
            running_loss += loss

        logger(f'Training loss for epoch {epoch}: {running_loss / len(train_loader)}')
        # Output metrics
        logger(f'Training accuracy for epoch {epoch}: {acc.compute()}')
        logger(f'Training precision for epoch {epoch}: {prec.compute()}')
        logger(f'Training recall for epoch {epoch}: {recall.compute()}')
        logger(f'Training F1-score for epoch {epoch}: {f1score.compute()}')
        logger('')

        # Reset metrics
        acc.reset()
        prec.reset()
        recall.reset()
        f1score.reset()

        model.eval()
        with torch.no_grad():
            for idx, batch in enumerate(val_loader):
                x, y = batch
                pred_y = model(x)

                loss = loss_fn(pred_y, y)
                acc.update(pred_y, y)
                prec.update(pred_y, y)
                recall.update(pred_y, y)
                f1score.update(pred_y, y)

                running_loss_val += loss

        logger(f'Validation loss for epoch {epoch}: {running_loss_val / len(val_loader)}')
        # Output metrics
        logger(f'Validation accuracy for epoch {epoch}: {acc.compute()}')
        logger(f'Validation precision for epoch {epoch}: {prec.compute()}')
        logger(f'Validation recall for epoch {epoch}: {recall.compute()}')
        logger(f'Validation F1-score for epoch {epoch}: {f1score.compute()}')

        # Log loss to W&B
        wandb.log({
            'epoch': epoch,
            'train_loss': running_loss / len(train_loader),
            'val_loss': running_loss_val / len(val_loader),
            'val_acc': acc.compute(),
            'val_precision': prec.compute(),
            'val_recall': recall.compute(),
            'val_f1score': f1score.compute()
            })
        
        # Reset metrics
        acc.reset()
        prec.reset()
        recall.reset()
        f1score.reset()

        logger(f'Finished epoch {epoch}')
        logger('')


    test_loader = DataLoader(test_dataset, batch_size = 4, shuffle = False)

    running_loss_test = 0.
    model.eval()

    # Start test loop
    for _, batch in enumerate(test_loader):
        x, y = batch
        pred_y = model(x)

        loss = loss_fn(pred_y, y)
        acc.update(pred_y, y)
        prec.update(pred_y, y)
        recall.update(pred_y, y)
        f1score.update(pred_y, y)

        running_loss_test += loss
        
    logger(f'Loss on testing data is {running_loss_test / len(test_loader)}')
    logger(f'Test metrics:')
    logger(f'Accuracy: {acc.compute()}')
    logger(f'Precision: {prec.compute()}')
    logger(f'Recall: {recall.compute()}')
    logger(f'F1 Score: {f1score.compute()}')

    # Log loss/metrics to W&B
    wandb.log({
        'test_loss': running_loss_test / len(test_loader),
        'test_acc': acc.compute(),
        'test_precision': prec.compute(),
        'test_recall': recall.compute(),
        'test_f1score': f1score.compute()
    })

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