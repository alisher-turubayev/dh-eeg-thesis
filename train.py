import logging
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC

import xgboost as xgb

def train_loop(loader, model, cv_folds, cv_repetitions):
    pass

def fit(loader, model_name, cv_folds, cv_repetitions):
    pca = PCA()
    if model_name == 'svm':
        model = LinearSVC(C = 10)
    else:
        model = xgb.XGBClassifier()
    pipe = Pipeline(steps = [('pca', pca), ('model', model)])

    for fold in range(cv_folds):
        logging.info(f'Starting fold {fold + 1} of {cv_folds}...')
        
        X, y = loader.get_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)

        for rep in range(cv_repetitions):
            pipe.fit(X = X_train, y = y_train)
            logging.info(f'Score of {model_name}: {model.score(X = X_test, y = y_test)}')
            logging.info(f'Repetition {rep + 1} complete.')

        logging.info(f'Fold {fold + 1} complete.\n')

    return