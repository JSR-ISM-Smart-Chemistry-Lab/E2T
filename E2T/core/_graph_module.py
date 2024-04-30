"""
"""

import torch
from ane.util.crystal import n_elem_feats as N_ELEM_FEATS
from ane.materials_property_prediction.gnn import get_gnn

from torch import nn
from torch_geometric.data import Batch, Dataset
from torch_geometric.loader import DataLoader
from pymatgen.core.structure import Structure

from ._module import LightningEpisodicModule


class LtGraphMNNs(LightningEpisodicModule):
    """
    """
    def __init__(
        self,
        gnn_model,
        emb_size,
        normalize_emb,
        header,
        loss=None,
        **kwargs,
    ):
        super(LtGraphMNNs, self).__init__(**kwargs)
        if loss is None:
            loss = torch.nn.MSELoss(reduction="mean")
        self.loss = loss
        self.gnn_model = gnn_model
        self.emb_size = emb_size
        self.normalize_emb = normalize_emb
        self.header = header

        if self.normalize_emb:
            self.layer_norm = nn.LinearNorm(self.emb_size)
            self.encoder = nn.Sequential(
                get_gnn(N_ELEM_FEATS, n_edge_feats=128, dim_out=self.emb_size, gnn_model=self.gnn_model),
                self.layer_norm,
            )
        else:
            self.encoder = get_gnn(N_ELEM_FEATS, n_edge_feats=128, dim_out=self.emb_size, gnn_model=self.gnn_model)

        self.save_hyperparameters(logger=False)

    def meta_learn(self, batch):
        self.encoder.train()
        support, query = batch
        s_emb = self.encoder(support)
        q_emb = self.encoder(query)
        y_hat = self.header(s_emb, support.y, q_emb)
        return self.loss(y_hat, query.y)

    def predict(self, support, query, scaler=None):
        self.encoder.eval()
        with torch.no_grad():
            s_emb = self.encoder(support)
            q_emb = self.encoder(query)

        y_hat = self.header(s_emb, support.y, q_emb)

        if scaler is not None:
            y_hat = scaler.inverse_transform(y_hat.cpu())

        return y_hat
