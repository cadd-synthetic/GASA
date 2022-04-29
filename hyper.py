from hyperopt import hp

gasa_hyperparameters = {
    'lr': hp.choice('lr', [0.001, 0.0006, 0.0008]),
    'batch_size': hp.choice('batch_size', [256, 128]),
    'hidden_dim1': hp.choice('hidden_dim1', [256, 200, 128]),
    'hidden_dim2': hp.choice('hidden_dim2', [64, 32]),
    'hidden_dim3': hp.choice('hidden_dim3', [64, 3232]),
    'num_heads': hp.choice('num_heads', [4, 6, 8]),
    'dropout': hp.choice('dropout', [0.3, 0.35, 0.1])}

def init_hyper_space(model):
    candidate_hypers = dict()
    if model == 'GASA':
        candidate_hypers.update(gasa_hyperparameters)
    else:
        return ValueError('Unexpected model: {}'.format(model))
    return candidate_hypers


class EarlyStopping:
    def __init__(self, patience=10,  mode='higher', metric='val_acc', filename=None):
        if filename is None:
            dt = datetime.datetime.now()
            filename = 'gasa_model_{}_{:02d}_{:02d}_{:02d}.pth'.format(dt.date(), dt.hour, dt.minute, dt.second)
        if metric is not None:
            assert metric in ['val_acc', 'val_loss', 'roc_auc_score'], \
                "Expect metric to be 'acc' or 'val_loss' or 'roc_auc_score', got {}".format(metric)
            if metric in ['val_acc', 'roc_auc_score']:
                print('For metric {}, the higher the better'.format(metric))
                mode = 'higher'
            if metric in ['val_loss']:
                print('For metric {}, the lower the better'.format(metric))
                mode = 'lower'
        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.filename = filename

    def _check_higher(self, score, prev_best_score):
        return score > prev_best_score

    def _check_lower(self, score, prev_best_score):
        return score < prev_best_score

    def step(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, model):
        torch.save({'model_state_dict': model.state_dict()}, self.filename)

    def load_checkpoint(self, model):
        model.load_state_dict(torch.load(self.filename)['model_state_dict'])

