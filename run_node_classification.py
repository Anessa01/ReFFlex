from attrdict import AttrDict
from torch_geometric.datasets import WebKB, WikipediaNetwork, Actor, Planetoid
from torch_geometric.utils import to_networkx, from_networkx, to_undirected
from torch_geometric.transforms import LargestConnectedComponents, ToUndirected
from experiments.node_classification import Experiment
import torch
import numpy as np
import pandas as pd
from hyperparams import get_args_from_input
from preprocessing import rewiring, sdrf, fosr, ReFFlex, digl, borf

torch.manual_seed(22)
np.random.seed(22)

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
datasets =  {"cora": cora, "citeseer": citeseer, "pubmed": pubmed}

for key in datasets:
    dataset = datasets[key]
    dataset.data.edge_index = to_undirected(dataset.data.edge_index)

def log_to_file(message, filename="results/node_classification.txt"):
    print(message)
    file = open(filename, "a")
    file.write(message)
    file.close()

default_args = AttrDict({
    "dropout": 0.5,
    "num_layers": 3,
    "hidden_dim": 128,
    "learning_rate": 1e-3,
    "layer_type": "R-GCN",
    "display": True,
    "num_trials": 100,
    "eval_every": 1,
    "rewiring": "borf",
    "num_iterations": 50,
    "num_relations": 2,
    "patience": 100,
    "dataset": None,
    "kappa" : 0.15, 
    "borf_batch_add" : 4,
    "borf_batch_remove" : 2
    })


results = []
args = default_args
args += get_args_from_input()
SMVList = np.zeros(args.num_layers+1, dtype=np.float64)

if args.dataset:
    # restricts to just the given dataset if this mode is chosen
    name = args.dataset
    datasets = {name: datasets[name]}

for key in datasets:
    accuracies = []
    best_test_accuracies = []
    print(f"TESTING: {key} ({default_args.rewiring})")
    dataset = datasets[key]
    if args.rewiring == "fosr":
        edge_index, edge_type, _ = fosr.edge_rewire(dataset.data.edge_index.numpy(), num_iterations=args.num_iterations)
        dataset.data.edge_index = torch.tensor(edge_index)
        dataset.data.edge_type = torch.tensor(edge_type)
    elif args.rewiring == "sdrf":
        dataset.data.edge_index, dataset.data.edge_type = sdrf.sdrf(dataset.data, loops=args.num_iterations, remove_edges=False, is_undirected=True)
    #print(rewiring.spectral_gap(to_networkx(dataset.data, to_undirected=True)))
    elif args.rewiring == "digl":
        for i in range(len(dataset)):
            dataset.data.edge_index = digl.rewire(dataset.data, alpha=0.1, eps=0.01)
            m = dataset.data.edge_index.shape[1]
            dataset.data.edge_type = torch.tensor(np.zeros(m, dtype=np.int64))
    elif args.rewiring == 'ReFFlex':
        edge_index, edge_type, _ = ReFFlex.rewiring_v3(dataset.data.edge_index.numpy(), beta=1, kappa=args.kappa)
        dataset.data.edge_index = torch.tensor(edge_index)
        dataset.data.edge_type = torch.tensor(edge_type)
    elif args.rewiring == "borf":
        print(f"[INFO] BORF hyper-parameter : num_iterations = {args.num_iterations}")
        print(f"[INFO] BORF hyper-parameter : batch_add = {args.borf_batch_add}")
        print(f"[INFO] BORF hyper-parameter : batch_remove = {args.borf_batch_remove}")
        dataset.data.edge_index, dataset.data.edge_type = borf.borf3(dataset.data, 
                loops=args.num_iterations, 
                remove_edges=False, 
                is_undirected=True,
                batch_add=args.borf_batch_add,
                batch_remove=args.borf_batch_remove,
                dataset_name=key,
                graph_index=0)
        print(len(dataset.data.edge_type))
    for trial in range(args.num_trials):
        #print(f"TRIAL {trial+1}")
        print(f"Trial {trial}/{args.num_trials}")
        train_acc, validation_acc, test_acc, SMV, best_test_acc = Experiment(args=args, dataset=dataset).run()
        SMVList = SMVList + SMV
        accuracies.append(test_acc)
        best_test_accuracies.append(best_test_acc)

    
    best_test_mean = 100 * np.mean(best_test_accuracies)
    best_test_ci = 200 * np.std(best_test_accuracies)/(args.num_trials ** 0.5)
    
    log_to_file(f"RESULTS FOR {key} ({default_args.rewiring}):\n")
    log_to_file(f"average acc: {np.mean(accuracies)}\n")
    log_to_file(f"plus/minus:  {2 * np.std(accuracies)/(args.num_trials ** 0.5)}\n\n")
    log_to_file(f"average best acc: {best_test_mean}+-{best_test_ci}\n\n")
    log_to_file(f"SMV: {SMVList / args.num_trials}")
    
    results.append({
        "dataset": key,
        "rewiring": args.rewiring,
        "num_iterations": args.num_iterations,
        "avg_accuracy": np.mean(accuracies),
        "ci":  2 * np.std(accuracies)/(args.num_trials ** 0.5),
        "SMV": SMVList / args.num_trials,
        "bt_ci": best_test_ci,
        "bt_acc": best_test_mean
        })
    results_df = pd.DataFrame(results)
    with open('results/node_classification.csv', 'a') as f:
        results_df.to_csv(f, mode='a', header=f.tell()==0)
