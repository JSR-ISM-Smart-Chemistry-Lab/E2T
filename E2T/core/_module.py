
import torch
from lightning.pytorch import LightningModule
from torch import nn


class LightningEpisodicModule(LightningModule):
    def __init__(
        self,
        lr: float = 1e-3,
        scheduler_step: int = 0,
        scheduler_decay: float = 0,
    ):
        super().__init__()
        self.lr = lr
        self.scheduler_step = scheduler_step
        self.scheduler_decay = scheduler_decay

    def training_step(self, batch, batch_idx):
        train_loss = self.meta_learn(batch)
        self.log(
            "train_loss",
            train_loss.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return train_loss

    def validation_step(self, batch, batch_idx):
        val_loss = self.meta_learn(batch)
        self.log(
            "val_loss",
            val_loss.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        return val_loss.item()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        if self.scheduler_step and self.scheduler_decay:
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.scheduler_step,
                gamma=self.scheduler_decay,
            )
            return [optimizer], [lr_scheduler]
        else:
            return optimizer


class LtMNNs(LightningEpisodicModule):
    """
    Family of Matching Neural Networks model with pytorch-lightning.

    Parameters
    ----------
    encoder : torch.nn.Module
        The encoder network.
    header : torch.nn.Module
        The header network.
    loss : torch.nn.Module, default=None
        The loss function to be used.
    """
    def __init__(
        self,
        encoder,
        header,
        loss=None,
        **kwargs,
    ):
        super(LtMNNs, self).__init__(**kwargs)
        if loss is None:
            loss = torch.nn.MSELoss(reduction="mean")
        self.loss = loss
        self.encoder = encoder
        self.header = header
        self.save_hyperparameters(logger=False)

    def meta_learn(self, batch):
        self.encoder.train()
        (support_x, _, support_y), (query_x, _, query_y) = batch
        s_emb = self.encoder(support_x)
        q_emb = self.encoder(query_x)
        y_hat = self.header(s_emb, support_y, q_emb)
        return self.loss(y_hat, query_y)

    def predict(self, support_x, support_y, query_x, scaler=None):
        """
        Predict the output of the query set.

        Parameters
        ----------
        support_x : torch.Tensor
            The support set input data.
        support_y : torch.Tensor
            The support set output data.
        query_x : torch.Tensor
            The query set input data.
        scaler : sklearn.preprocessing.StandardScaler, default=None
            The scaler to be used to inverse transform the output.
        """
        self.encoder.eval()
        with torch.no_grad():
            s_emb = self.encoder(support_x)
            q_emb = self.encoder(query_x)

        y_hat = self.header(s_emb, support_y.reshape(-1, 1), q_emb)

        if scaler is not None:
            y_hat = scaler.inverse_transform(y_hat.cpu())

        return y_hat


class RidgeRegressionHeader(nn.Module):
    """
    Ridge regression header for E2T.

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength.
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.
    """
    def __init__(self, alpha=1.0, fit_intercept=True):
        super(RidgeRegressionHeader, self).__init__()
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def forward(self, support_x, support_y, query_x):
        """
        Parameters
        ----------
        support_x : torch.Tensor
            The support set input data.
        support_y : torch.Tensor
            The support set output data.
        query_x : torch.Tensor
            The query set input data.
        """
        device = support_x.device

        if self.fit_intercept:
            support_x = self._add_intercept(support_x, device)
            query_x = self._add_intercept(query_x, device)

        covariance_matrix = torch.matmul(support_x.transpose(-1, -2), support_x) + self.alpha * torch.eye(support_x.shape[1]).to(device)
        precision_matrix = torch.linalg.pinv(covariance_matrix)

        y_hat = torch.matmul(
            torch.matmul(query_x, precision_matrix),
            torch.matmul(support_x.transpose(-1, -2), support_y)
        )

        return y_hat

    @staticmethod
    def _add_intercept(x, device):
        shape = list(x.shape)
        shape[-1] = 1
        return torch.cat([x, torch.ones(shape, device=device)], dim=-1)


class FCEncoder(nn.Module):
    def __init__(
        self,
        sizes,
        dropout=0.2,
        batch_norm=False,
        normalize_out=True,
    ):
        super().__init__()
        self.sizes = sizes
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.normalize_out = normalize_out

        # dropout preprocess
        if isinstance(dropout, float):
            dropouts = [dropout] * (len(sizes) - 2)
        elif isinstance(dropout, (tuple, list)):
            assert len(dropouts) == (len(sizes) - 2)
            dropouts = dropout

        # definition of network
        layers = []
        for i in range(1, len(sizes)):
            layers.append(nn.Linear(sizes[i-1], sizes[i]))
            if i < len(sizes) - 1:
                # Batch normalization
                if batch_norm:
                    layers.append(nn.BatchNorm1d(sizes[i]))
                # Activation function
                layers.append(nn.ReLU())
                # Dropout
                if dropouts[i-1] > 0:
                    layers.append(nn.Dropout(dropouts[i-1]))

        if normalize_out:
            layers.append(nn.LayerNorm(sizes[i]))

        self.layers = nn.Sequential(*layers)
        self.initialize_weights()

    def initialize_weights(self):
        """ initialize wieghts using He's method"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, X):
        emb = self.layers(X)
        return emb
