import torch

from jaffalearn import CheckpointHandler, Engine
from jaffalearn.data import DataLoaderFactory, MixedDataLoaderFactory

import system.models as models
import system.utils as utils
from system import Baseline


def predict(subset,
            epochs,
            checkpoint_dir,
            sample_rate=None,
            block_length=None,
            features=None,
            cache_features=False,
            weights_path=None,
            batch_size=1,
            partition=None,
            n_workers=0,
            cuda=True,
            seed=None,
            ):
    # Determine which device (CPU or GPU) to use
    device = utils.determine_device(cuda)

    # Create data loader for subset
    if partition:
        factory = MixedDataLoaderFactory(
            block_lengths=partition['block_lengths'],
            batch_sizes=partition['batch_sizes'],
            sample_rate=sample_rate,
            features=features,
            n_workers=n_workers,
            cache_features=cache_features,
            labeler=None,
        )
    else:
        factory = DataLoaderFactory(
            sample_rate=sample_rate,
            block_length=block_length,
            features=features,
            batch_size=batch_size,
            n_workers=n_workers,
            cache_features=cache_features,
            labeler=None,
        )
    loader = factory.test_data_loader(subset)

    # Load checkpoints (possibly lazily)
    checkpoint_handler = CheckpointHandler(checkpoint_dir)
    if weights_path is not None and weights_path.name:
        checkpoints = [torch.load(weights_path, map_location=device)]
    else:
        checkpoints = (checkpoint_handler.load(epoch=epoch, device=device)
                       for epoch in epochs)

    # Generate predictions for each checkpoint
    y_preds = []
    for checkpoint in checkpoints:
        system = Baseline.from_checkpoint(checkpoint, models.create_model)
        engine = Engine(system, device)

        y_preds.append(engine.predict(loader, show_progress=True))

    y_pred = torch.stack(y_preds).mean(dim=0)
    y_pred = y_pred.cpu().numpy()
    return y_pred
