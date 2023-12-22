import gin
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils import perclass_accuracy
import wandb

@gin.configurable('CNNClassifier')
class CNNClassifier(pl.LightningModule):
    def __init__(
        self,
        input_size,
        output_size,
        window_datapoints,
        hidden_size,
        kernel_size,
        stride,
        maxpool_kernel_size,
        nn_size,
        dropout_rate,
        fold_idx
    ):
        super().__init__()
        self.idx = fold_idx
        layers = []

        if window_datapoints is not None:
            # If window datapoints is passed, we assume that data is signal data and windowed
            curr_len = window_datapoints

            # First layer - 1D Convolution with Dropout layer
            layers.append(nn.Conv1d(input_size, hidden_size, kernel_size, stride))
            
        else:
            # Otherwise, we assume the data is in 4 segments (as per Medeiros et al (2021), p. 12)
            # Thus, our calculations change a bit
            # Data is of shape [batch size, segments, # of features]
            curr_len = input_size # where input_size is the # of features per segment
            layers.append(nn.Conv1d(4, hidden_size, kernel_size, stride))
            
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Calculate new length - for formula see https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        curr_len = int((curr_len - (kernel_size - 1) - 1) / stride + 1)
        
        # Second layer - Max pooling
        layers.append(nn.MaxPool1d(maxpool_kernel_size))
        
        # Calculate new length - for formula see https://pytorch.org/docs/stable/generated/torch.nn.MaxPool1d.html
        curr_len = int((curr_len - (maxpool_kernel_size - 1) - 1) / maxpool_kernel_size + 1)
        
        # Third layer - fully connected NN with ReLU -> reduce to nn_size
        # Flatten size is size of convolution multiplied by current length of sequence
        layers.append(nn.Flatten())
        layers.append(nn.Linear(hidden_size * curr_len, nn_size))

        # Final layer - fully connected NN + Softmax activation layer
        layers.append(nn.Linear(nn_size, output_size))
        layers.append(nn.Softmax(dim = 1))
        
        self.model = nn.Sequential(*layers)
        self.loss_fn = F.cross_entropy

    def training_step(self, batch, _):
        x, y = batch
        # Move to device (because the tensor by default is on CPU)
        x = x.to(self.device)
        y = y.to(self.device)

        y_pred = self.model(x)
        loss = self.loss_fn(y_pred, y)

        self.log('train_step_loss', loss)
        return loss
    
    def validation_step(self, batch, _):
        x, y = batch
        # Move to device (because the tensor by default is on CPU)
        x = x.to(self.device)
        y = y.to(self.device)
        
        y_pred = self.model(x)
        loss = self.loss_fn(y_pred, y)

        self.log('val_step_loss', loss)
        return loss

    def test_step(self, batch, _):
        x, y = batch
        # Move to device (because the tensor by default is on CPU)
        x = x.to(self.device)
        y = y.to(self.device)

        y_pred = self.model(x)
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