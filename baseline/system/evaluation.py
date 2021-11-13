import pandas as pd
import torch
import torchmetrics.functional as metrics


def evaluate(y_pred, y_true, label_set):
    # Compute class-wise, macro-averaged, and micro-average scores
    y_pred = torch.as_tensor(y_pred, dtype=torch.float32)
    y_true = torch.as_tensor(y_true, dtype=torch.int32)
    class_scores = compute_scores(y_pred, y_true, average=None)
    macro_scores = class_scores.mean(dim=0, keepdims=True)
    micro_scores = compute_scores(y_pred, y_true, average='micro')

    # Crete DataFrame for scores
    scores = torch.cat((class_scores, macro_scores, micro_scores)).numpy()
    index = label_set + ['Macro Average', 'Micro Average']
    columns = ['Average Precision', 'Accuracy']
    return pd.DataFrame(scores, pd.Index(index, name='Class'), columns)


def compute_scores(y_pred, y_true, average=None):
    kwargs = {
        'num_classes': y_true.shape[1],
        'average': average,
    }
    ap = metrics.average_precision(y_pred, y_true, **kwargs)
    acc = accuracy(y_pred, y_true, **kwargs)

    # Return scores as a tensor with shape (N, 2)
    return torch.tensor([ap, acc]).T.view(-1, 2)


def accuracy(y_pred, y_true, **kwargs):
    return metrics.accuracy(y_pred, y_true.argmax(dim=1), **kwargs)
