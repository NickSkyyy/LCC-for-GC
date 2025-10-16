import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import TransformerEncoder, TransformerEncoderLayer

class CMPX(nn.Module):
  def __init__(self, args, dropout, feature_size, hidden_size, num_ATT_layers, num_classes, num_GNN_layers):
    super(CMPX, self).__init__()
    self.name = "CMPX"
    self.feature_size = feature_size
    self.hidden_size = hidden_size
    self.num_classes = num_classes
    self.num_ATT_layers = num_ATT_layers
    self.num_GNN_layers = num_GNN_layers

    # [origin, graphs, line_graphs]
    self.origin_layers = nn.ModuleList()
    for _ in range(self.num_GNN_layers):
      encoder = TransformerEncoderLayer(d_model=self.feature_size, nhead=1, dim_feedforward=self.hidden_size, dropout=0.5)
      self.origin_layers.append(TransformerEncoder(encoder, self.num_ATT_layers))
    cnt = 1
    if args.node_cmp == 1:
      cnt += 1
      self.graph_layers = nn.ModuleList()
      for _ in range(self.num_GNN_layers):
        encoder = TransformerEncoderLayer(d_model=self.feature_size, nhead=1, dim_feedforward=self.hidden_size, dropout=0.5)
        self.graph_layers.append(TransformerEncoder(encoder, self.num_ATT_layers))
    if args.edge_info == 1 or args.edge_cmp == 1:
      cnt += 1
      self.lg_layers = nn.ModuleList()
      for _ in range(self.num_GNN_layers):
        encoder = TransformerEncoderLayer(d_model=self.feature_size, nhead=1, dim_feedforward=self.hidden_size, dropout=0.5)
        self.lg_layers.append(TransformerEncoder(encoder, self.num_ATT_layers))

    self.predictions = nn.Linear(self.feature_size * self.num_GNN_layers * cnt, self.num_classes)
    self.dropouts = nn.Dropout(dropout)

  def forward(self, origin_x, origin_gp, origin_fea, graph_x, graph_gp, graph_fea, lg_x, lg_gp, lg_fea):
    output = 0
    output_layer = [[] for _ in range(3)]

    origin_temp = F.embedding(origin_x, origin_fea)
    graph_temp = None
    lg_temp = None
    if graph_x is not None:
      graph_temp = F.embedding(graph_x, graph_fea)
    if lg_x is not None:
      lg_temp = F.embedding(lg_x, lg_fea)

    for idx in range(self.num_GNN_layers):
      origin_out = self.origin_layers[idx](origin_temp)[0]
      output_layer[0].append(origin_out)
      origin_temp = F.embedding(origin_x, origin_out)
      if graph_x is not None:
        graph_out = self.graph_layers[idx](graph_temp)[0]
        output_layer[1].append(graph_out)
        graph_temp = F.embedding(graph_x, graph_out)
      if lg_x is not None:
        lg_out = self.lg_layers[idx](lg_temp)[0]
        output_layer[2].append(lg_out)
        lg_temp = F.embedding(lg_x, lg_out)
    origin_temp = torch.cat(output_layer[0], 1)
    origin_graph_temp = torch.spmm(origin_gp, origin_temp)
    if graph_x is not None:
      graph_temp = torch.cat(output_layer[1], 1)
      graph_graph_temp = torch.spmm(graph_gp, graph_temp)
      origin_graph_temp = torch.cat((origin_graph_temp, graph_graph_temp), 1)
    if lg_x is not None:
      lg_temp = torch.cat(output_layer[2], 1)
      lg_graph_temp = torch.spmm(lg_gp, lg_temp)
      origin_graph_temp = torch.cat((origin_graph_temp, lg_graph_temp), 1)

    origin_graph_temp = self.dropouts(origin_graph_temp)
    output = self.predictions(origin_graph_temp)
    return output, origin_temp, graph_temp

class U2GNN(nn.Module):
  def __init__(self, dropout, feature_size, hidden_size, num_ATT_layers, num_classes, num_GNN_layers):
    """
    num_classes: number of graph classes
    num_ATT_layers: number of attention layers
    num_GNN_layers: number of GNN layers
    """
    super(U2GNN, self).__init__()
    self.name = "U2GNN"
    self.feature_size = feature_size
    self.hidden_size = hidden_size
    self.num_classes = num_classes
    self.num_ATT_layers = num_ATT_layers
    self.num_GNN_layers = num_GNN_layers
    
    self.layers = nn.ModuleList()
    for _ in range(self.num_GNN_layers):
      encoder = TransformerEncoderLayer(d_model=self.feature_size, nhead=1, dim_feedforward=self.hidden_size, dropout=0.5)
      self.layers.append(TransformerEncoder(encoder, self.num_ATT_layers))

    self.predictions = nn.Linear(self.feature_size * self.num_GNN_layers, self.num_classes)
    self.dropouts = nn.Dropout(dropout)
    self.softmax = nn.LogSoftmax(1)

  def forward(self, input, graph_pool, concat):
    output = 0
    output_layer = []
    # embeddings return the shape of (input x feature_size)
    input_temp = F.embedding(input, concat)
    for idx in range(self.num_GNN_layers):
      output_temp = self.layers[idx](input_temp)[0]
      output_layer.append(output_temp)
      input_temp = F.embedding(input, output_temp)
    output_temp = torch.cat(output_layer, 1)
    graph_temp = torch.spmm(graph_pool, output_temp)
    graph_temp = self.dropouts(graph_temp)
    output = self.predictions(graph_temp)
    return output