import torch

from jaffalearn import CheckpointHandler, Engine, Logger
from jaffalearn.data import DataLoaderFactory, MixedDataLoaderFactory

import system.models as models
import system.utils as utils
from system import Baseline, LabelNoise


def train(log_dir,
          checkpoint_dir,
          train_set,
          val_set,
          sample_rate=None,
          block_length=None,
          hop_length=None,
          features=None,
          cache_features=False,
          model='vgg11a',
          weights_path=None,
          label_noise=None,
          n_epochs=1,
          batch_size=1,
          lr=0.001,
          lr_scheduler=None,
          partition=None,
          cuda=True,
          n_workers=0,
          seed=None,
          overwrite=False,
          ):
    if seed is not None:
        utils.ensure_reproducibility(seed)

    # Instantiate model/system and engine
    model_args = {
        'model_name': model,
        'n_channels': train_set.dataset.n_channels,
        'n_classes': len(train_set.dataset.label_set),
    }
    model = models.create_model(**model_args)
    system = Baseline(model, lr, lr_scheduler, model_args)
    device = utils.determine_device(cuda)
    engine = Engine(system, device)

    # Add callbacks for logging and checkpoint handling
    logger = Logger(system, log_dir, overwrite)
    checkpoint_handler = CheckpointHandler(checkpoint_dir, engine)
    engine.callbacks += [logger, checkpoint_handler]

    # Restore training if there is a checkpoint to continue from
    if len(logger.history) > 0:
        checkpoint = checkpoint_handler.load(logger.history, device=device)
        engine.restore_state(checkpoint)
        logger.truncate(engine.epoch)
    elif weights_path is not None and weights_path.name:
        system.restore_state(torch.load(weights_path, map_location=device))

    # Create data loaders for training and validation
    if partition:
        factory = MixedDataLoaderFactory(
            block_lengths=partition['block_lengths'],
            batch_sizes=partition['batch_sizes'],
            sample_rate=sample_rate,
            features=features,
            n_workers=n_workers,
            cache_features=cache_features,
        )
    else:
        factory = DataLoaderFactory(
            sample_rate=sample_rate,
            block_length=block_length,
            hop_length=hop_length,
            features=features,
            batch_size=batch_size,
            n_workers=n_workers,
            cache_features=cache_features,
        )
    loader_val = factory.validation_data_loader(val_set)
    if label_noise:
        factory.labeler = LabelNoise(**label_noise).relabeler(train_set)
    loader_train = factory.training_data_loader(train_set)

    print('\n', model, '\n', sep='')

    engine.fit(loader_train, loader_val, n_epochs, show_progress=True)
