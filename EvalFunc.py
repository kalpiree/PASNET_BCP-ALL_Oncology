import torch

from sklearn.metrics import roc_auc_score, f1_score


def auc(y_true, y_pred):
    ###covert one-hot encoding into integer
    # y_true = torch.argmax(y_true, dim=1)
    ###if gpu is being used, transferring back to cpu
    if torch.cuda.is_available():
        y_true = y_true.cpu().detach()
        y_pred = y_pred.cpu().detach()
    ###
    test1 = y_pred.detach().numpy()
    test2 = y_true.detach().numpy()
    auc = roc_auc_score(y_true.detach().numpy(), y_pred.detach().numpy(), multi_class='ovo')
    return (auc)


def f1(y_true, y_pred):
    ###covert one-hot encoding into integer
    y = torch.argmax(y_true, dim=1)
    ###estimated targets (either 0 or 1)
    pred = torch.argmax(y_pred, dim=1)
    ###if gpu is being used, transferring back to cpu
    if torch.cuda.is_available():
        y = y.cpu().detach()
        pred = pred.cpu().detach()
    ###
    a = y.numpy()
    b = pred.numpy()
    # f1 = f1_score(y.detach().numpy(), pred.detach().numpy(), average='macro')
    f1 = f1_score(y.detach().numpy(), pred.detach().numpy(), average='macro')
    return (f1)
