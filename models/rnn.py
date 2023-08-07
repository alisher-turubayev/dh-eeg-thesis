import gin
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics.classification as mtr
import pytorch_lightning as pl

@gin.configurable('RNNClassifier')
class RNNClassifier(pl.LightningModule):
    def __init__(
        self,
        input_size,
        output_size,
        window_size,
        hidden_size = gin.REQUIRED,
        n_layers = gin.REQUIRED
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
            nn.Linear(n_layers * hidden_size, output_size),
            nn.Softmax(dim = 1)
        )

        # Determine task 
        if output_size == 1:
            metrics_task = "binary"
            self.loss_fn = F.binary_cross_entropy
        else:
            metrics_task = "multiclass"
            self.loss_fn = F.cross_entropy

        # Register metrics
        self.acc = mtr.Accuracy(task = metrics_task, num_classes = output_size, average = 'macro')
        self.prec = mtr.Precision(task = metrics_task, num_classes = output_size, average = 'macro')
        self.recall = mtr.Recall(task = metrics_task, num_classes = output_size, average = 'macro')
        self.f1score = mtr.F1Score(task = metrics_task, num_classes = output_size, average = 'macro')

    def training_step(self, batch, _):
        x, y = batch
        # Move to device (because the tensor by default is on CPU)
        x = x.to(self.device)
        y = y.to(self.device)
        # Transpose x to conform to RNN input (to [batch_size, num_observations, num_channels)
        x = torch.transpose(x, dim0 = 1, dim1 = 2)
        # Set the state to None - this way, RNN layer will autocreate the hidden state
        h = None
        
        steps = x.shape[1]
        for i in range(steps - 1):
            # Select a row, then add back one dimension to comply with RNN input [batch_size, 1, num_channels]
            x_t = torch.unsqueeze(torch.select(x, dim = 1, index = i), dim = 1)
            _, h = self.model(x_t, h)
        # Transpose from [n_layers, batch_size, hidden_size] to [batch_size, n_layers, hidden_size]
        h = torch.transpose(h, 0, 1)
        # Remove the second dimension
        h = torch.flatten(h, 1, 2)
        # Because we only need the last output (many-to-one RNN), we only take output at the very end
        pred_y = self.activation(h)

        loss = self.loss_fn(pred_y, y)

        self.log('train_step_loss', loss)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        x = torch.transpose(x, dim0 = 1, dim1 = 2)
        h = None

        steps = x.shape[1]
        for i in range(steps - 1):
            x_t = torch.unsqueeze(torch.select(x, dim = 1, index = i), dim = 1)
            _, h = self.model(x_t, h)
        h = torch.transpose(h, 0, 1)
        h = torch.flatten(h, 1, 2)
        pred_y = self.activation(h)

        loss = self.loss_fn(pred_y, y)
        self.acc(pred_y, y)
        
        self.log('val_step_loss', loss)
        self.log('val_acc', self.acc)
        return loss

    def on_validation_end(self):
        self.acc.reset()
        return super().on_validation_end()

    def test_step(self, batch, _):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        x = torch.transpose(x, dim0 = 1, dim1 = 2)
        h = None

        steps = x.shape[1]
        for i in range(steps - 1):
            x_t = torch.unsqueeze(torch.select(x, dim = 1, index = i), dim = 1)
            _, h = self.model(x_t, h)
        h = torch.transpose(h, 0, 1)
        h = torch.flatten(h, 1, 2)
        pred_y = self.activation(h)

        # Update metrics
        self.prec(pred_y, y)
        self.recall(pred_y, y)
        self.f1score(pred_y, y)

        # Log metrics
        self.log('precision', self.prec)
        self.log('recall', self.recall)
        self.log('f1score', self.f1score)

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr = 0.001, weight_decay = 0.01)
