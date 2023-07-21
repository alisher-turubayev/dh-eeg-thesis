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

from models.cnn import CNNClassifier
from models.rnn import RNNClassifier

import xgboost as xgb

@gin.configurable('train_params')
def train_test_loop(dataset, model_name, epochs, cv_folds, cv_repetitions, lr = 0.001, weight_decay = 0.01):
    train_dataset, val_dataset = random_split(dataset, [0.7, 0.3])

    train_loader = DataLoader(train_dataset, batch_size = 4, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = 4, shuffle = False)

    if model_name == 'cnn':
        model = CNNClassifier()
    else: 
        model = RNNClassifier()

    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)

    loss_fn = F.cross_entropy

    logging.info('--------------------')
    logging.info('Starting training...')

    for epoch in range(epochs):
        running_loss = 0.
        running_loss_val = 0.

        for _, batch in enumerate(train_loader):
            x, y = batch

            optimizer.zero_grad()
            pred_y = model(x)

            loss = loss_fn(y, pred_y)
            loss.backward()

            optimizer.step()
            running_loss += loss

        logging.info(f'Training loss for epoch {epoch}: {running_loss / len(train_loader)}')

        model.eval()
        with torch.no_grad():
            for _, batch in enumerate(val_loader):
                x, y = batch
                pred_y = model(x)

                loss = loss_fn(y, pred_y)

                running_loss_val += loss

        logging.info(f'Validation loss for epoch {epoch}: {running_loss_val / len(val_loader)}')
        logging.info(f'Finished epoch {epoch}')
        logging.info('')

    return

def fit(dataset, model_name, _):
    pca = PCA()
    if model_name == 'svm':
        model = LinearSVC(C = 10)
    else:
        model = xgb.XGBClassifier()
    pipe = Pipeline(steps = [('pca', pca), ('model', model)])

    X, y = dataset.get_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)

    pipe.fit(X = X_train, y = y_train)
    logging.info(f'Score of {model_name}: {model.score(X = X_test, y = y_test)}')
    
    return