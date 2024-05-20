"""
Test rewired GNN performance on graph classifiation benchmarks.
"""

from attrdict import AttrDict
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx, from_networkx, to_dense_adj
from torch_geometric.data import Data
from experiments.graph_classification import Experiment
import torch
import numpy as np
import pandas as pd
from hyperparams import get_args_from_input
from preprocessing import rewiring, sdrf, fosr, digl, ReFFlex
import os
import networkx as nx
import wandb
import scipy.sparse as sp
import time

wandb.init(
            # set the wandb project where this run will be logged
            project="ReFFlex-GCN",
    
            # track hyperparameters and run metadata
            config={
            "learning_rate": 0.02,
            "architecture": "rewiring",
            "dataset": "Gnp",
            "epochs": 0,
            }
        )

torch.manual_seed(22)

for n in range(6800, 10000, 100):
    #G = nx.gnp_random_graph(n, 2*np.log(n)/n)
    G = nx.gnp_random_graph(n, 1/n)
    adj = sp.coo_matrix(nx.to_numpy_matrix(G))
    edge_index = np.vstack((adj.row, adj.col))
    pygdata = Data(x=np.array([0]*n), y=np.array([0]*n), edge_index=torch.LongTensor(edge_index))
    
    timestart = time.time()
    _ = fosr.edge_rewire(edge_index, num_iterations=20)
    timeend = time.time()
    wandb.log({"fosr": (timeend - timestart)})
    
    timestart = time.time()
    _ = sdrf.sdrf(pygdata, loops=20)
    timeend = time.time()
    wandb.log({"sdrf": (timeend - timestart)})
    
    timestart = time.time()
    _, _, runtime = ReFFlex.rewiring_v3(edge_index, kappa=1.,beta=1.)
    timeend = time.time()
    wandb.log({"ReFFlex": runtime})
    




mutag = list(TUDataset(root="data", name="MUTAG"))
enzymes = list(TUDataset(root="data", name="ENZYMES"))
proteins = list(TUDataset(root="data", name="PROTEINS"))
collab = list(TUDataset(root="data", name="COLLAB"))
imdb = list(TUDataset(root="data", name="IMDB-BINARY"))
reddit = list(TUDataset(root="data", name="REDDIT-BINARY"))
datasets = {"mutag": mutag, "enzymes": enzymes, "proteins": proteins, "collab": collab, "reddit": reddit, "imdb": imdb}
#datasets = {"imdb": imdb}
#datasets = {"proteins": proteins, "collab": collab}
save = False
for key in datasets:
    if key in ["reddit", "imdb", "collab"]:
        for graph in datasets[key]:
            n = graph.num_nodes
            graph.x = torch.ones((n,1))

def average_spectral_gap(dataset):
    # computes the average spectral gap out of all graphs in a dataset
    spectral_gaps = []
    for graph in dataset:
        G = to_networkx(graph, to_undirected=True)
        spectral_gap = rewiring.spectral_gap(G)
        spectral_gaps.append(spectral_gap)
    return sum(spectral_gaps) / len(spectral_gaps)

def log_to_file(message, filename="results/graph_classification.txt"):
    print(message)
    file = open(filename, "a")
    file.write(message)
    file.close()

default_args = AttrDict({
    "dropout": 0.5,
    "num_layers": 4,
    "hidden_dim": 64,
    "learning_rate": 1e-3,
    "layer_type": "R-GCN",
    "display": True,
    "num_trials": 100,
    "eval_every": 1,
    "rewiring": "ReFFlex",
    "num_iterations": 20,
    "patience": 50,
    "output_dim": 2,
    "alpha": 0.1,
    "eps": 0.001,
    "dataset": None,
    "last_layer_fa": False
    })

hyperparams = {
    "mutag": AttrDict({"output_dim": 2}),
    "enzymes": AttrDict({"output_dim": 6}),
    "proteins": AttrDict({"output_dim": 2}),
    "collab": AttrDict({"output_dim": 3}),
    "imdb": AttrDict({"output_dim": 2}),
    "reddit": AttrDict({"output_dim": 2})
}

results = []
args = default_args
args += get_args_from_input()
if args.dataset:
    # restricts to just the given dataset if this mode is chosen
    name = args.dataset
    datasets = {name: datasets[name]}

for key in datasets:
    args += hyperparams[key]
    train_accuracies = []
    validation_accuracies = []
    test_accuracies = []
    energies = []
    print(f"TESTING: {key} ({args.rewiring})")
    dataset = datasets[key]
    if args.rewiring == "fosr":
        for i in range(len(dataset)):
            edge_index, edge_type, _ = fosr.edge_rewire(dataset[i].edge_index.numpy(), num_iterations=args.num_iterations)
            dataset[i].edge_index = torch.tensor(edge_index)
            dataset[i].edge_type = torch.tensor(edge_type)
    elif args.rewiring == "sdrf":
        for i in range(len(dataset)):
            dataset[i].edge_index, dataset[i].edge_type = sdrf.sdrf(dataset[i], loops=args.num_iterations, remove_edges=False, is_undirected=True)
    elif args.rewiring == "digl":
        for i in range(len(dataset)):
            dataset[i].edge_index = digl.rewire(dataset[i], alpha=0.1, eps=0.05)
            m = dataset[i].edge_index.shape[1]
            dataset[i].edge_type = torch.tensor(np.zeros(m, dtype=np.int64))
    elif args.rewiring == 'ReFFlex':
        for i in range(len(dataset)):
            print(f"{i}/{len(dataset)}")
            if save:
                dirFile_edge_index = "savedRewiring/ReFFlex_{}/{}_edge_index.npy".format(key, i)
                dirFile_edge_type = "savedRewiring/ReFFlex_{}/{}_edge_type.npy".format(key, i)
                if os.path.exists(dirFile_edge_index):
                    edge_index = np.load(dirFile_edge_index)
                    edge_type = np.load(dirFile_edge_type)
                else:
                    edge_index, edge_type = ReFFlex.rewiring_v3(dataset[i].edge_index.numpy(), beta=1, kappa=1.)
                    np.save(dirFile_edge_index, edge_index)
                    np.save(dirFile_edge_type, edge_type)
                    print("Saved.")
            else:
                edge_index, edge_type = ReFFlex.rewiring_v3(dataset[i].edge_index.numpy(), beta=1, kappa=1.)
            dataset[i].edge_index = torch.tensor(edge_index)
            dataset[i].edge_type = torch.tensor(edge_type)
    #spectral_gap = average_spectral_gap(dataset)
    for trial in range(args.num_trials):
        print(f"Trial {trial}/{args.num_trials}")
        train_acc, validation_acc, test_acc, energy = Experiment(args=args, dataset=dataset).run()
        train_accuracies.append(train_acc)
        validation_accuracies.append(validation_acc)
        test_accuracies.append(test_acc)
        energies.append(energy)
    train_mean = 100 * np.mean(train_accuracies)
    val_mean = 100 * np.mean(validation_accuracies)
    test_mean = 100 * np.mean(test_accuracies)
    energy_mean = 100 * np.mean(energies)
    train_ci = 200 * np.std(train_accuracies)/(args.num_trials ** 0.5)
    val_ci = 200 * np.std(validation_accuracies)/(args.num_trials ** 0.5)
    test_ci = 200 * np.std(test_accuracies)/(args.num_trials ** 0.5)
    energy_ci = 200 * np.std(energies)/(args.num_trials ** 0.5)
    log_to_file(f"RESULTS FOR {key} ({args.rewiring}), {args.num_iterations} ITERATIONS:\n")
    log_to_file(f"average acc: {test_mean}\n")
    log_to_file(f"plus/minus:  {test_ci}\n\n")
    results.append({
        "dataset": key,
        "rewiring": args.rewiring,
        "layer_type": args.layer_type,
        "num_iterations": args.num_iterations,
        "alpha": args.alpha,
        "eps": args.eps,
        "test_mean": test_mean,
        "test_ci": test_ci,
        "val_mean": val_mean,
        "val_ci": val_ci,
        "train_mean": train_mean,
        "train_ci": train_ci,
        "energy_mean": energy_mean,
        "energy_ci": energy_ci,
        "last_layer_fa": args.last_layer_fa
        })
df = pd.DataFrame(results)
with open('results/graph_classification_fa.csv', 'a') as f:
    df.to_csv(f, mode='a', header=f.tell()==0)
