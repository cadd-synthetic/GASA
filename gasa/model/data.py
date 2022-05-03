import os
import dgl
import json
from torch.utils.data import Dataset

class pred_data(Dataset):
    def __init__(self, graph, smiles):
        self.smiles = smiles
        self.graph = graph
        self.lens = len(smiles)

    def __len__(self):
        return self.lens

    def __getitem__(self, item):
        return self.smiles[item], self.graph[item]


def predict_collate(samples):
    smiles, graphs = map(list, zip(*samples))
    bg = dgl.batch(graphs)
    return smiles, bg


def mkdir_p(path):
    try:
        os.makedirs(path)
        print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            print('Directory {} already exists.'.format(path))
        else:
            raise

def init_trial_path(args):
    trial_id = 0
    path_exists = True
    while path_exists:
        trial_id += 1
        path_to_results = args['result_path'] + '/{:d}'.format(trial_id)
        path_exists = os.path.exists(path_to_results)
    args['trial_path'] = path_to_results
    mkdir_p(args['trial_path'])

    return args


def get_configure(model):
    path = os.getcwd()
    p = os.path.join(path, "model/gasa.json")
    with open(p, 'r') as f:
        config = json.load(f)
    return config
