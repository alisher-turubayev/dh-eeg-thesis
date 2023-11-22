import argparse
import numpy as np
from sklearn.metrics import confusion_matrix

def add_args(parser: argparse.ArgumentParser):
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
        choices = ['medeiros', 'medeiros_raw', 'custom', 'custom_raw'],
        required = True,
        help = 'Dataset to use for experiment. `raw` suffix denotes datasets with raw data for DL models, such as CNN/RNN'
    )
    parser.add_argument(
        '--pca_cev',
        type = float,
        help = 'PCA: Target CEV to select the number of components. See sklearn.decomposition.PCA for details'
    )
    # SVM parameters
    parser.add_argument(
        '--svm_dual',
        action = 'store_true',
        help = 'SVM: whether to use dual or primal optimization problem. Specify to use dual optimization problem. See sklearn.svm.LinearSVC for details'
    )
    parser.add_argument(
        '--svm_C',
        type = float,
        help = 'SVM: regularization parameter. See sklearn.svm.LinearSVC for details'
    )
    parser.add_argument(
        '--svm_max_iter',
        type = int,
        help = 'SVM: number of iterations to train the SVM. See sklearn.svm.LinearSVC for details'
    )
    # XGBoost parameters
    parser.add_argument(
        '--xgb_eval_metric',
        choices = ['auc', 'aucpr', 'merror'],
        help = 'XGBoost: evaluation metric for XGBoost. See xgboost.XGBClassifier for details'
    )
    parser.add_argument(
        '--xgb_objective',
        choices = ['softmax', 'softprob'],
        help = 'XGBoost: the loss function to minimize. See xgboost.XGBClassifier for details'
    )
    parser.add_argument(
        '--xgb_max_depth',
        type = int,
        help = 'XGBoost: maximum tree depth to use. See xgboost.XGBClassifier for details'
    )
    parser.add_argument(
        '--xgb_tree_method',
        choices = ['auto', 'hist', 'gpu_hist', 'exact', 'approx'],
        help = 'XGBoost: tree construction algorithm to use. See xgboost.XGBClassifier for details'
    )
    # RNN parameters
    parser.add_argument(
        '--rnn_hidden_size',
        type = int,
        help = 'RNN: hidden size of the RNN network'
    )
    parser.add_argument(
        '--rnn_n_layers',
        type = int,
        help = 'RNN: number of hidden layers of the RNN network'
    )
    # CNN parameters
    parser.add_argument(
        '--cnn_hidden_size',
        type = int,
        help = 'CNN: the hidden size of the convolution layer'
    )
    parser.add_argument(
        '--cnn_kernel_size',
        type = int,
        help = 'CNN: the kernel size of the convolution layer. See torch.nn.Conv1d for details'
    )
    parser.add_argument(
        '--cnn_stride',
        type = int,
        help = 'CNN: the stride of the convolution layer. See torch.nn.Conv1d for details'
    )
    parser.add_argument(
        '--cnn_maxpool_kernel_size',
        type = int,
        help = 'CNN: size of the maxpool kernel size'
    )
    parser.add_argument(
        '--cnn_nn_size',
        type = int,
        help = 'CNN: size of the fully-connected layer'
    )
    parser.add_argument(
        '--cnn_dropout_rate',
        type = float,
        help = 'CNN: the dropout rate to use in fully-connected neural network before Softmax'
    )

def validate_args(args: dict[str, any]):
    # Check that raw dataset is not used with ML models
    assert not ((args['model'] == 'svm' or args['model'] == 'xgboost') and ('raw' in args['dataset'])), 'Cannot use `raw` dataset with ML models'
    # Check that all parameters are present for SVM
    if args['model'] == 'svm':
        assert args.get('pca_cev') is not None      \
            and args.get('pca_cev') >= 0.0          \
            and args.get('pca_cev') <= 1.0, 'The PCA CEV parameter must be between 0.0 and 1.0'
        assert args.get('svm_dual') is not None     \
            and args.get('svm_C') is not None       \
            and args.get('svm_max_iter') is not None, 'Need to specify all SVM parameters'
    # Check that all parameters are present for XGBoost
    elif args['model'] == 'xgboost':
        assert args.get('pca_cev') is not None      \
            and args.get('pca_cev') >= 0.0          \
            and args.get('pca_cev') <= 1.0, 'The PCA CEV parameter must be between 0.0 and 1.0'
        assert args.get('xgb_eval_metric') is not None  \
            and args.get('xgb_objective') is not None   \
            and args.get('xgb_max_depth') is not None   \
            and args.get('xgb_tree_method') is not None, 'Need to specify all XGBoost parameters'
    # Check that all parameters are present for RNN
    elif args['model'] == 'rnn':
        assert args.get('rnn_hidden_size') is not None \
            and args.get('rnn_n_layers') is not None, 'Need to specify all XGBoost parameters'
    # Check that all parameters are present for CNN
    else:
        assert args.get('cnn_hidden_size') is not None          \
            and args.get('cnn_kernel_size') is not None         \
            and args.get('cnn_stride') is not None              \
            and args.get('cnn_maxpool_kernel_size') is not None \
            and args.get('cnn_nn_size') is not None             \
            and args.get('cnn_dropout_rate') is not None, 'Need to specify all XGBoost parameters'

# Adapted from:
# https://stackoverflow.com/a/65673016
def perclass_accuracy(y, y_pred, class_pos):
    cm = confusion_matrix(y, y_pred)
    tn = np.sum(np.delete(np.delete(cm, class_pos, axis = 0), class_pos, axis = 1))
    tp = cm[class_pos, class_pos]

    return (tp + tn) / np.sum(cm)