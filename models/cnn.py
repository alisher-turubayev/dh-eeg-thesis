import gin
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics.classification as mtr
import pytorch_lightning as pl

@gin.configurable('CNNClassifier')
class CNNClassifier(pl.LightningModule):
    def __init__(
        self,
        input_size,
        output_size,
        window_size,
        n_layers = gin.REQUIRED,
        kernel_size = gin.REQUIRED,
        hidden_size = gin.REQUIRED,
    ):
        super().__init__()
        layers = []
        # TODO: fix warning on padding = 'same'
        layers.append(nn.Conv1d(input_size, hidden_size, kernel_size, padding = 'same'))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool1d(int(kernel_size / 2)))
        layers.append(nn.Flatten())
        # TODO: this is a mess
        # Initially, kernel size is 200
        # Divide by 2 -> 100
        # Because pooling -> window size / kernel size => 2000 / 100 = 20 (so our maxpool layer has output size of 20)
        # Then, when we flatten we essentially have an output shape of (batch_size, hidden_size * maxpool_out_size)
        flatten_size = int(hidden_size * (window_size / (kernel_size / 2)))
        layers.append(nn.Linear(flatten_size, output_size))
        layers.append(nn.Softmax())

        # TODO: create logic to handle tabular data (i.e. window_size is NaN)
        self.model = nn.Sequential(*layers)

        # Determine task 
        if output_size == 1:
            metrics_task = "binary"
            self.loss_fn = F.binary_cross_entropy
        else:
            metrics_task = "multiclass"
            self.loss_fn = F.cross_entropy

        # Register metrics
        # TODO: check if this works
        self.acc = mtr.Accuracy(task = metrics_task, num_classes = output_size, average = 'macro')
        self.prec = mtr.Precision(task = metrics_task, num_classes = output_size, average = 'macro')
        self.recall = mtr.Recall(task = metrics_task, num_classes = output_size, average = 'macro')
        self.f1score = mtr.F1Score(task = metrics_task, num_classes = output_size, average = 'macro')

    def training_step(self, batch, _):
        x, y = batch
        pred_y = self.model(x)
        loss = self.loss_fn(pred_y, y)

        self.log('train_step_loss', loss)
        return loss
    
    def validation_step(self, batch, _):
        x, y = batch
        pred_y = self.model(x)
        loss = self.loss_fn(pred_y, y)
        self.acc(pred_y, y)

        self.log('val_step_loss', loss)
        self.log('val_step_acc', self.acc)
        return loss
    
    def on_validation_end(self):
        self.acc.reset()
        return super().on_validation_end()

    def test_step(self, batch, _):
        x, y = batch
        pred_y = self.model(x)
        
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