import gin
import torch.nn as nn

@gin.configurable('CNNClassifier')
class CNNClassifier(nn.Module):
    def __init__(
        self,
        input_size,
        window_size,
        n_layers,
        kernel_size,
        hidden_size, 
        output_size
    ):
        super().__init__()
        layers = []
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

        self.model = nn.Sequential(*layers)

    def forward(self, batch):
        return self.model(batch)