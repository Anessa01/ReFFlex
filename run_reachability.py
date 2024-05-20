from attrdict import AttrDict
from torch_geometric.datasets import WebKB, WikipediaNetwork, Actor, Planetoid
from torch_geometric.utils import to_networkx, from_networkx, to_undirected
from torch_geometric.transforms import LargestConnectedComponents, ToUndirected
from experiments.node_classification import Experiment
import torch
import numpy as np
import pandas as pd
from hyperparams import get_args_from_input
from preprocessing import rewiring, sdrf, fosr, ReFFlex, digl

largest_cc = LargestConnectedComponents()


cornell = WebKB(root="data", name="Cornell")
wisconsin = WebKB(root="data", name="Wisconsin")
texas = WebKB(root="data", name="Texas")
chameleon = WikipediaNetwork(root="data", name="chameleon")
squirrel = WikipediaNetwork(root="data", name="squirrel")
actor = Actor(root="data")
cora = Planetoid(root="data", name="cora")
citeseer = Planetoid(root="data", name="citeseer")
pubmed = Planetoid(root="data", name="pubmed")
datasets = {"cornell": cornell, "wisconsin": wisconsin, "texas": texas, "chameleon": chameleon, "squirrel": squirrel, "actor": actor, "cora": cora, "citeseer": citeseer, "pubmed": pubmed}
#datasets = {"cornell": cornell, "wisconsin": wisconsin}

for key in datasets:
    dataset = datasets[key]
    dataset.data.edge_index = to_undirected(dataset.data.edge_index)

default_args = AttrDict({
    "dropout": 0.5,
    "num_layers": 3,
    "hidden_dim": 128,
    "learning_rate": 1e-3,
    "layer_type": "R-GCN",
    "display": True,
    "num_trials": 100,
    "eval_every": 1,
    "rewiring": "ReFFlex",
    "num_iterations": 50,
    "num_relations": 2,
    "patience": 100,
    "dataset": None,
    "kappa" : 0.15
    })


results = []
args = default_args
args += get_args_from_input()
SMVList = np.zeros(args.num_layers+1, dtype=np.float64)

for key in datasets:
    accuracies = []
    best_test_accuracies = []
    print(f"TESTING: {key} ({default_args.rewiring})")
    dataset = datasets[key]
    if args.rewiring == 'ReFFlex':
        ReFFlex.BORF_reachability2(dataset.data.edge_index.numpy(), beta=1, kappa=args.kappa)
