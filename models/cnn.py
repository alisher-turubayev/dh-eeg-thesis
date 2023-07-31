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
        window_datapoints,
        hidden_size = gin.REQUIRED,
        kernel_size = gin.REQUIRED,
        stride = gin.REQUIRED,
        maxpool_kernel_size = gin.REQUIRED,
        nn_size = gin.REQUIRED,
        dropout_rate = gin.REQUIRED
    ):
        super().__init__()
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
        # Determine task 
        if window_datapoints is None:
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

        pred_y = self.model(x)
        loss = self.loss_fn(pred_y, y)

        self.log('train_step_loss', loss)
        return loss
    
    def validation_step(self, batch, _):
        x, y = batch
        # Move to device (because the tensor by default is on CPU)
        x = x.to(self.device)
        y = y.to(self.device)
        
        pred_y = self.model(x)
        loss = self.loss_fn(pred_y, y)
        # Calculate accuracy on validation - we use this metric to early stop training
        self.acc(pred_y, y)

        self.log('val_step_loss', loss)
        self.log('val_acc', self.acc, on_step = False, on_epoch = True)
        return loss
    
    def on_validation_end(self):
        self.acc.reset()
        return super().on_validation_end()

    def test_step(self, batch, _):
        x, y = batch
        # Move to device (because the tensor by default is on CPU)
        x = x.to(self.device)
        y = y.to(self.device)

        pred_y = self.model(x)
        
        # Update metrics of interest
        self.prec(pred_y, y)
        self.recall(pred_y, y)
        self.f1score(pred_y, y)

        # Log metrics
        self.log('precision', self.prec)
        self.log('recall', self.recall)
        self.log('f1score', self.f1score)

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr = 0.001, weight_decay = 0.01)