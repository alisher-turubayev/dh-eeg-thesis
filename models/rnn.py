import gin
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils import perclass_accuracy
import wandb

@gin.configurable('RNNClassifier')
class RNNClassifier(pl.LightningModule):
    def __init__(
        self,
        input_size,
        output_size,
        window_size,
        hidden_size,
        n_layers,
        fold_idx
    ):
        super().__init__()
        # Use keyed arguments instead of positional arguments
        # NOTE: 1-to-1 output
        self.model = nn.RNN(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = n_layers,
            batch_first = True
        )
        self.activation = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim = 1)
        )
        self.loss_fn = F.mse_loss
        self.idx = fold_idx

    def training_step(self, batch, _):
        x, y = batch
        # Move to device (because the tensor by default is on CPU)
        x = x.to(self.device)
        y = y.to(self.device)
        # Transpose x to conform to RNN input (to [batch_size, num_observations, num_channels])
        x = torch.transpose(x, dim0 = 1, dim1 = 2)
        # Set the state to None - this way, RNN layer will autocreate the hidden state
        h = None
        # Feed the data into the model
        out, h = self.model(x, h)
        # We are only interested in last output for each batch item, so we take that and discard the rest
        out = torch.select(out, dim = 1, index = (out.shape[1] - 1))
        # Pass the RNN output through the activation layer (fully connected layer + softmax)
        y_pred = self.activation(out)

        loss = self.loss_fn(y_pred, y)

        self.log('train_step_loss', loss)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        x = torch.transpose(x, dim0 = 1, dim1 = 2)
        h = None

        out, h = self.model(x, h)
        out = torch.select(out, dim = 1, index = (out.shape[1] - 1))
        y_pred = self.activation(out)

        loss = self.loss_fn(y_pred, y)

        self.log('val_step_loss', loss)
        return loss

    def test_step(self, batch, _):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        x = torch.transpose(x, dim0 = 1, dim1 = 2)
        h = None

        out, h = self.model(x, h)
        y_pred = self.activation(out)

        # Transform y_pred from probabilities to actual prediction
        y_pred = torch.argmax(y_pred, dim = 1)
        y = torch.argmax(y, dim = 1)

        wandb.log({
            'fold_acc': accuracy_score(y, y_pred),
            'fold_acc_class0': perclass_accuracy(y, y_pred, class_pos = 0),
            'fold_acc_class1': perclass_accuracy(y, y_pred, class_pos = 1),
            'fold_acc_class2': perclass_accuracy(y, y_pred, class_pos = 2),
            'fold_acc_class3': perclass_accuracy(y, y_pred, class_pos = 3),
            'fold_precision': precision_score(y, y_pred, average = 'micro'),
            'fold_recall': recall_score(y, y_pred, average = 'micro'),
            'fold_f1_score': f1_score(y, y_pred, average = 'micro')
        })
    
    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr = 0.001, weight_decay = 0.01)
