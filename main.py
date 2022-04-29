import json
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from data import pred_data, mkdir_p, init_trial_path, get_configure, predict_collate
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, recall_score, precision_score,  precision_recall_curve
import pandas as pd
from hyper import init_hyper_space, EarlyStopping
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from copy import deepcopy
from gasa_utils import generate_graph
from classifier import gasa_classifier


def run_train_epoch(args, epoch, model, train_loader, loss_func, optimizer):
    model.train()
    train_loss, acc = 0
    pred_all = []
    label_all = []
    for iter, (smiles, bg, label) in enumerate(train_loader):
        bg, label = bg.to(DEVICE), label.to(DEVICE)
        prediction = model(bg)
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
    train_losses.append(train_loss)
    train_acces.append(train_acc)
    fpr, tpr, thresholds = roc_curve(label_all, pred_all, pos_label=0)
    return train_loss, train_acces, train_acc, fpr, tpr, thresholds


def run_val_epoch(args, model, val_loader, loss_func):
    model.eval()
    val_pred = []
    val_label = []
    pos_pro = []
    neg_pro = []
    with torch.no_grad():
        for iter, (smiles, bg) in enumerate(val_loader):
            pred = torch.softmax(model(bg), 1)
            pos_pro += pred[:, 0].detach().cpu().numpy().tolist()
            neg_pro += pred[:, 1].detach().cpu().numpy().tolist()
            pred1 = torch.max(pred, 1)[1].view(-1)  
            val_pred += pred1.detach().cpu().numpy().tolist()
    return val_pred, pos_pro, neg_pro


def Find_Optimal_Cutoff(TPR, FPR, threshold): 
    y = TPR - FPR
    an = np.argwhere(y == np.amax(y))
    Youden_index = an.flatten().tolist()
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point


def statistic(y_true, y_pred):
    c_mat = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tp, fn, fp, tn = list(c_mat.flatten())
    sp = tn / (tn + fp)
    return tp, fn, fp, tn, sp


def load_model(exp_configure):
    if exp_configure['model'] == 'GASA':
        model = gasa_classifier(dropout=exp_configure['dropout'], 
                                num_heads=exp_configure['num_heads'], 
                                hidden_dim1=exp_configure['hidden_dim1'], 
                                hidden_dim2=exp_configure['hidden_dim2'], 
                                hidden_dim3=exp_configure['hidden_dim3'])
    else:
        return ValueError("Expect model 'GASA', got{}".format((exp_configure['model'])))
    return model

def GASA(smiles, exp_config, args):
    torch.manual_seed(0)
    np.random.seed(0)
    exp_config.update({'model': args['model']}) 
    #save results 
    args = init_trial_path(args)  
    graph = generate_graph(smiles)
    data = pred_data(graph=graph, smiles=smiles)             
    data_loader = DataLoader(data, batch_size=exp_config['batch_size'], shuffle=False, collate_fn=predict_collate)
    model = load_model(exp_config)
    loss_func = nn.CrossEntropyLoss()
# train the model, if retrain the model, load data first 
    # optimizer = torch.optim.Adam(model.parameters(), lr=exp_config['lr'])
    # stopper = EarlyStopping(metric='roc_auc_score', mode='higher', patience=20, filename=args['trial_path'] + '/gasa.pth')
    # with open(args['trial_path'] + '/configure.json', 'w') as f:
    #     json.dump(exp_config, f, indent=2)
    
    # for epoch in range(args['num_epochs']):
    #     train_loss, _, train_acc, fpr, tpr, threshold = run_train_epoch(args, epoch, model, train_loader, loss_func, optimizer)
    #     val_pred, val_pos, val_neg = run_val_epoch(args, model, val_loader, loss_func)
    #     early_stop = stopper.step(train_loss, model)
    #     print('epoch {:d}/{:d}, train_acc {:.4f}, train_loss {:.4f}, bset_score {:.4f}'.format(epoch + 1, args['num_epochs'], train_acc, train_loss, stopper.best_score)
    #     if early_stop:
    #         print("Early stopping")
    #         break
    # stopper.load_checkpoint(model)
   
    checkpoint = torch.load('./gasa.pth', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    pred, pos, neg = run_val_epoch(args, model, data_loader, loss_func)
    return pred, pos, neg
    

#hyperparameter optmization
def bayesian_optimization(args):
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

    return best_trial_path

    
if __name__ == '__main__':
    from argparse import ArgumentParser
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
        exp_config = get_configure(args['model'])
        #load data path  
        df = pd.read_csv(r'./test.csv')
        smiles = df['smiles'].tolist()
        pred, pos, neg = GASA(smiles, exp_config, args)
        #print predict results
        print(pred, pos, neg)
        trial_path = args['result_path'] + '/1'
