import json
import torch
import os
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, recall_score, precision_score
import pandas as pd
from model.gasa_utils import generate_graph
from model.model import gasa_classifier
from model.data import pred_data, mkdir_p, init_trial_path, get_configure, predict_collate
from model.hyper import init_hyper_space, EarlyStopping
from hyperopt import fmin, tpe
from copy import deepcopy
from argparse import ArgumentParser

def parse():
    '''
    load Parameters
    '''
    parser = ArgumentParser(' Binary Classification')
    parser.add_argument('-n', '--num-epochs', type=int, default=80)
    parser.add_argument('-mo', '--model', default='GASA')
    parser.add_argument('-ne', '--num-evals', type=int, default=None,
                        help='the number of hyperparameter searches (default: None)')
    parser.add_argument('-me', '--metric', choices=['acc', 'loss', 'roc_auc_score'],
                        help='Metric for evaluation (default: roc_auc_score)')
    parser.add_argument('-p', '--result-path', type=str, default='gasa/results',
                        help='Path to save training results (default: classification_results)')
    args = parser.parse_args().__dict__

    if args['num_evals'] is not None:
        assert args['num_evals'] > 0, 'Expect the number of hyperparameter search trials to ' \
                                        'be greater than 0, got {:d}'.format(args['num_evals'])
        print('Start hyperparameter search with Bayesian '
                'optimization for {:d} trials'.format(args['num_evals']))
        trial_path = bayesian_optimization(args)
    else:
        print('Use the manually specified hyperparameters')

    return args


def run_train_epoch(args, epoch, model, train_loader, loss_func, optimizer):
    '''
    if retrain model
    Parameters
    epoch: Number of iterations
    model: the model for train
    train_loader: load the data for training
    loss_func: Loss function
    optimizer: The optimizer (Adam)
    return: 
    train_loss: compute loss
    train_acc: compute accurary
    '''
    model.train()
    pred_all = []
    label_all = []
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for iter, (smiles, bg, label) in enumerate(train_loader):
        bg, label = bg.to(DEVICE), label.to(DEVICE)
        prediction = model(bg)[0]
        loss = loss_func(prediction, label)
        optimizer.zero_grad()  
        loss.backward()  
        optimizer.step() 
        train_loss += loss.detach().item() 
        pred = torch.max(prediction, 1)[1]
        pred_all.extend(pred.cpu().numpy())
        label_all.extend(label.cpu().numpy())
        acc += pred.eq(label.view_as(pred)).cpu().sum()
    train_acc = acc.numpy() / len(train_loader.dataset)
    train_loss /= (iter + 1)
    return train_loss, train_acc


def run_val_epoch(args, model, val_loader, loss_func):
    model.eval()
    val_pred = []
    val_label = []
    pos_pro = []
    neg_pro = []
    with torch.no_grad():
        for iter, (smiles, bg) in enumerate(val_loader):
            pred = model(bg)[0]
            pos_pro += pred[:, 0].detach().cpu().numpy().tolist()
            neg_pro += pred[:, 1].detach().cpu().numpy().tolist()
            pred1 = torch.max(pred, 1)[1].view(-1)  
            val_pred += pred1.detach().cpu().numpy().tolist()
    return val_pred, pos_pro, neg_pro


def Find_Optimal_Cutoff(TPR, FPR, threshold): 
    '''
    Compute Youden index, find the optimal threshold
    Parameters
    TPR: True Positive Rate
    FPR: False Positive Rate
    threshold: 
    return
    optimal_threshold: optimal_threshold
    point: optimum coordinates
    '''
    y = TPR - FPR
    an = np.argwhere(y == np.amax(y))
    Youden_index = an.flatten().tolist()
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point


def statistic(y_true, y_pred):
    '''
    compute statistic results
    Parameters
    y_true: true label of the given molecules
    y_pred: predicted label 
    return
    tp: True Positive
    fn: False Negative
    fp: False Positive
    tn: True Negative
    '''
    c_mat = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tp, fn, fp, tn = list(c_mat.flatten())
    return tp, fn, fp, tn


def load_model(exp_configure):
    '''
    load GASA model
    '''
    if exp_configure['model'] == 'GASA':
        model = gasa_classifier(dropout=exp_configure['dropout'], 
                                num_heads=exp_configure['num_heads'], 
                                hidden_dim1=exp_configure['hidden_dim1'], 
                                hidden_dim2=exp_configure['hidden_dim2'], 
                                hidden_dim3=exp_configure['hidden_dim3'])
    else:
        return ValueError("Expect model 'GASA', got{}".format((exp_configure['model'])))
    return model


def GASA(smiles):
    '''
    GASA model for prediction
    Parameters
    smiles: SMILES representation of the moelcule of interest
    return
    pred: predicted label for the given molecules (0:ES; 1:HS)
    pos: the postive probability
    neg: the negative probability 
    '''
    args = parse()
    torch.manual_seed(0)
    np.random.seed(0)
    exp_config = get_configure(args['model'])
    exp_config.update({'model': args['model']}) 
    args = init_trial_path(args)
    ls_smi = []
    if isinstance(smiles, list):  
        ls_smi = smiles
    else:
        ls_smi.append(smiles)
    graph = generate_graph(ls_smi)
    data = pred_data(graph=graph, smiles=ls_smi)             
    data_loader = DataLoader(data, batch_size=exp_config['batch_size'], shuffle=False, collate_fn=predict_collate)
    model = load_model(exp_config)
    loss_func = nn.CrossEntropyLoss()
    path = os.getcwd()
    pth = os.path.join(path, "model/gasa.pth")
    checkpoint = torch.load(pth, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    trial_path = args['result_path'] + '/1'
    pred, pos, neg = run_val_epoch(args, model, data_loader, loss_func)
    return pred, pos, neg
    


def bayesian_optimization():
    '''
    hyperparameter optmization
    '''
    args = parse()
    results = []
    candidate_hypers = init_hyper_space(args['model'])

    def objective(hyperparams):
        configure = deepcopy(args)
        trial_path, val_metric = main(configure, hyperparams)

        if args['metric'] in ['roc_auc_score', 'val_acc']:
            val_metric_to_minimize = 1 - val_metric 
        else:
            val_metric_to_minimize = val_metric
        results.append((trial_path, val_metric_to_minimize))
        return val_metric_to_minimize

    fmin(objective, candidate_hypers, algo=tpe.suggest, max_evals=args['num_evals'])
    results.sort(key=lambda tup: tup[1])
    best_trial_path, best_val_metric = results[0]

    return best_val_metric