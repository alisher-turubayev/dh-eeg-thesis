from pytorch_lightning import LightningModule


class CNNClassifier(LightningModule):
    def __init__(
        self      
    ):
        super().__init__()

    def forward(self, batch, batch_idx):
        pass