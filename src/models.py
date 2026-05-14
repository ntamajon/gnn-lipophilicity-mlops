import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    GCNConv,
    GINConv,
    SAGEConv,
    GATConv,
    global_mean_pool,
)
from torch_geometric.nn import GCNConv, global_mean_pool, GINConv, global_mean_pool
from torch_geometric.nn import SAGEConv, GATConv
from torch_geometric.nn import JumpingKnowledge
from torch_geometric.nn import AttentionalAggregation


class GCNGraph(nn.Module):
    def __init__(self, num_features, num_classes=1,
                 hidden_channels=64, n_layers=3, dropout=0.2):
        super().__init__()

        self.n_layers = n_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        # Primera capa
        self.convs.append(GCNConv(num_features, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Capas intermedias
        for _ in range(n_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Cabeza final (regresión)
        self.lin = nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch

        # Message passing
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Pooling global
        x = global_mean_pool(x, batch)

        # Regresión final
        x = self.lin(x)

        return x.view(-1)


class GINMean(nn.Module):
    def __init__(self, in_dim, hidden=128, layers=5, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(layers):
            mlp = nn.Sequential(
                nn.Linear(in_dim if i == 0 else hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(hidden))

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        g = global_mean_pool(x, batch)
        return self.head(g).view(-1)
    

class SAGeMean(nn.Module):
    def __init__(self, in_dim, hidden=128, layers=4, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(SAGEConv(in_dim, hidden))
        self.bns.append(nn.BatchNorm1d(hidden))
        for _ in range(layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        g = global_mean_pool(x, batch)
        return self.head(g).view(-1)
    

class GATMean(nn.Module):
    def __init__(self, in_dim, hidden=64, heads=4, layers=3, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # Primera capa
        self.convs.append(GATConv(in_dim, hidden, heads=heads, dropout=dropout))
        self.bns.append(nn.BatchNorm1d(hidden * heads))

        for _ in range(layers - 1):
            self.convs.append(GATConv(hidden * heads, hidden, heads=heads, dropout=dropout))
            self.bns.append(nn.BatchNorm1d(hidden * heads))

        self.head = nn.Sequential(
            nn.Linear(hidden * heads, hidden * heads),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * heads, 1)
        )

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        g = global_mean_pool(x, batch)
        return self.head(g).view(-1)

class GCN_JK_Att(nn.Module):
    def __init__(self, in_dim, hidden=64, layers=4, dropout=0.2, jk_mode="cat"):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(GCNConv(in_dim, hidden))
        self.bns.append(nn.BatchNorm1d(hidden))
        for _ in range(layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))

        self.jk = JumpingKnowledge(mode=jk_mode)
        jk_out = hidden * layers if jk_mode == "cat" else hidden

        # Pooling atencional
        self.pool = AttentionalAggregation(gate_nn=nn.Linear(jk_out, 1))

        self.head = nn.Sequential(
            nn.Linear(jk_out, jk_out),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(jk_out, 1)
        )

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        xs = []
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xs.append(x)

        h = self.jk(xs)
        g = self.pool(h, batch)
        return self.head(g).view(-1)

def make_model(model_name, dataset):
    if model_name == "GCN_baseline":
        return GCNGraph(
            num_features=dataset.num_features,
            num_classes=1,
            hidden_channels=64,
            n_layers=3,
            dropout=0.2
        )

    elif model_name == "GIN_mean":
        return GINMean(in_dim=dataset.num_features, hidden=128, layers=5, dropout=0.2)

    elif model_name == "SAGE_mean":
        return SAGeMean(in_dim=dataset.num_features, hidden=128, layers=4, dropout=0.2)

    elif model_name == "GAT_mean":
        return GATMean(in_dim=dataset.num_features, hidden=64, heads=4, layers=3, dropout=0.2)

    elif model_name == "GCN_JK_Att":
        return GCN_JK_Att(in_dim=dataset.num_features, hidden=64, layers=4, dropout=0.2, jk_mode="cat")

    else:
        raise ValueError(f"Unknown model_name: {model_name}")