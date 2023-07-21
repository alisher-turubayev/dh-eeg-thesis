import gin
import torch.nn as nn

@gin.configurable('RNNClassifier')
class RNNClassifier(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        nonlinearity = 'relu'
        ):
        super().__init__()
        self.model = nn.RNN(input_size, hidden_size, nonlinearity = nonlinearity)
        self.activation = nn.Linear(hidden_size, output_size)

    def forward(self, batch, _):
        h = self.model(batch)
        return self.activation(h)
