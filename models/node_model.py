import torch
import torch.nn as nn
from torch.nn import ModuleList, Dropout, ReLU
from torch_geometric.nn import GCNConv, RGCNConv, SAGEConv, GINConv, FiLMConv, global_mean_pool

class RGINConv(torch.nn.Module):
    def __init__(self, in_features, out_features, num_relations):
        super(RGINConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_relations = num_relations
        self.self_loop_conv = torch.nn.Linear(in_features, out_features)
        convs = []
        for i in range(self.num_relations):
            convs.append(GINConv(nn.Sequential(nn.Linear(in_features, out_features),nn.BatchNorm1d(out_features), nn.ReLU(),nn.Linear(out_features, out_features))))
        self.convs = ModuleList(convs)
    def forward(self, x, edge_index, edge_type):
        x_new = self.self_loop_conv(x)
        for i, conv in enumerate(self.convs):
            rel_edge_index = edge_index[:, edge_type==i]
            x_new += conv(x, rel_edge_index)
        return x_new

class GCN(torch.nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        self.args = args
        self.num_relations = args.num_relations
        self.layer_type = args.layer_type
        num_features = [args.input_dim] + list(args.hidden_layers) + [args.output_dim]
        self.num_layers = len(num_features) - 1
        layers = []
        for i, (in_features, out_features) in enumerate(zip(num_features[:-1], num_features[1:])):
            layers.append(self.get_layer(in_features, out_features))
        self.layers = ModuleList(layers)

        self.reg_params = list(layers[0].parameters())
        self.non_reg_params = list([p for l in layers[1:] for p in l.parameters()])

        self.dropout = Dropout(p=args.dropout)
        self.act_fn = ReLU()
    def get_layer(self, in_features, out_features):
        if self.layer_type == "GCN":
            return GCNConv(in_features, out_features)
        elif self.layer_type == "R-GCN":
            return RGCNConv(in_features, out_features, self.num_relations)
        elif self.layer_type == "GIN":
            return GINConv(nn.Sequential(nn.Linear(in_features, out_features),nn.BatchNorm1d(out_features), nn.ReLU(),nn.Linear(out_features, out_features)))
        elif self.layer_type == "R-GIN":
            return RGINConv(in_features, out_features, self.num_relations)
        elif self.layer_type == "SAGE":
            return SAGEConv(in_features, out_features)
        elif self.layer_type == "FiLM":
            return FiLMConv(in_features, out_features)
    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def _smoothness(self, h):
        numV = h.shape[0]
        
        print("\t[SMV] Calculating hNorm")
        hNorm = h / torch.norm(h, p=1, dim=1).unsqueeze(0).t()
        hNorm = torch.where(torch.isnan(hNorm), torch.full_like(hNorm, 0), hNorm)
        hNorm = hNorm.detach()
        SMV = 0
        for i in range(numV):
            #print(f"{i}/{numV}-{SMV}")
                SMV += torch.sum(torch.norm(hNorm[i] - hNorm, p=2, dim=1)).item()
        
        return SMV / numV / (numV - 1) / 2
    
    def forward(self, graph, measure_SMV=False):
        x, edge_index = graph.x, graph.edge_index
        
        if measure_SMV:
            SMVList = []
        for i, layer in enumerate(self.layers):
            if measure_SMV:
                # check smoothness value instead of computing final values
                SMV = self._smoothness(x)
                SMVList.append(SMV)
            if self.layer_type in ["R-GCN", "R-GIN"]:
                x = layer(x, edge_index, edge_type=graph.edge_type)
            else:
                x = layer(x, edge_index)
            if i != self.num_layers - 1:
                x = self.act_fn(x)
                x = self.dropout(x)
        if measure_SMV:
            # check SMV instead of computing final values
            return SMVList
        return x
