from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

import torchmetrics.functional as metrics

from jaffalearn import SupervisedSystem

import system.evaluation as evaluation


class Baseline(SupervisedSystem):
    def __init__(self, model, lr, lr_scheduler, model_args=None):
        eval_metrics = {
            'acc': evaluation.accuracy,
            'mAP': metrics.average_precision,
        }
        super().__init__(model, cce, metrics=eval_metrics,
                         model_args=model_args)

        # Use AdamW for optimization
        self._optimizer = AdamW(model.parameters(), lr)

        # Determine which LR scheduler to use
        method = lr_scheduler['method']
        if method == 'step':
            self._scheduler = StepLR(
                self._optimizer,
                lr_scheduler['step_size'],
                lr_scheduler['gamma'],
            )
        else:
            raise ValueError(f"Unknown scheduler method '{method}'")

        self.add_hyperparameters(lr=lr, lr_scheduler=lr_scheduler)

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def scheduler(self):
        return self._scheduler

    @classmethod
    def from_checkpoint(cls, checkpoint, model_fn):
        params = {k: v for k, v in checkpoint['hyperparameters'].items()
                  if k in ['lr', 'lr_scheduler', 'model_args']}
        model = model_fn(**params['model_args'])
        system = cls(model, **params)
        system.restore_state(checkpoint)
        return system


def cce(y_pred, y_true):
    return (-y_true * y_pred.log_softmax(dim=1)).sum(dim=1).mean()
