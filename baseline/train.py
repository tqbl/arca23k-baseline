import argparse
import json
import pprint
import sys
from pathlib import Path

import cli


def train(args):
    import system.datasets as datasets
    import system.training as training

    # Load the relevant dataset metadata
    dataset = datasets.dataset(args.dataset, args.arca23k_dir,
                               args.fsd50k_dir, args.params.seed)
    train_set = dataset['training']
    val_set = dataset['validation']

    if args.frac < 1:
        train_set = datasets.sample(train_set, args.frac, args.params.seed)

    # Ensure output directories exist
    log_dir = args.work_dir / args.experiment_id / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.work_dir / args.experiment_id / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Log parameters to json file and stdout
    params = vars(args.params)
    json_params = dict(params)
    json_params['weights_path'] = str(params['weights_path'])
    pprint.pprint(json_params, sort_dicts=False)
    with open(log_dir / 'parameters.json', 'w') as f:
        json.dump(json_params, f, indent=2)

    print()
    print(f'log_dir: {log_dir}')
    print(f'checkpoint_dir: {checkpoint_dir}')

    training.train(log_dir, checkpoint_dir, train_set, val_set, **params)


def parse_args():
    config, conf_parser, remaining_args = cli.parse_config_args()
    parser = argparse.ArgumentParser(parents=[conf_parser])

    parser.add_argument('dataset', metavar='DATASET')
    parser.add_argument('--experiment_id', metavar='ID')
    parser.add_argument('--work_dir', type=Path, metavar='DIR')
    parser.add_argument('--arca23k_dir', type=Path, metavar='DIR')
    parser.add_argument('--fsd50k_dir', type=Path, metavar='DIR')
    parser.add_argument('--frac', type=float, metavar='NUM')

    # Group the following command-line parameters
    params = cli.GroupParser('params', parser)
    params.add_argument('--sample_rate', type=cli.maybe_float, metavar='NUM')
    params.add_argument('--block_length', type=cli.maybe_float, metavar='NUM')
    params.add_argument('--hop_length', type=cli.maybe_float, metavar='NUM')
    params.add_argument('--features', type=cli.dict, metavar='SPEC')
    params.add_argument('--cache_features', type=cli.boolean, metavar='BOOL')
    params.add_argument('--model', choices=['vgg9a', 'vgg11a'])
    params.add_argument('--weights_path', type=Path, metavar='PATH')
    params.add_argument('--label_noise', type=cli.dict, metavar='DICT')
    params.add_argument('--n_epochs', type=int, metavar='N')
    params.add_argument('--batch_size', type=int, metavar='N')
    params.add_argument('--lr', type=float, metavar='NUM')
    params.add_argument('--lr_scheduler', type=cli.dict, metavar='SPEC')
    params.add_argument('--partition', type=cli.dict, metavar='SPEC')
    params.add_argument('--seed', type=cli.maybe_int, metavar='N')
    params.add_argument('--cuda', type=cli.boolean, metavar='BOOL')
    params.add_argument('--n_workers', type=int, metavar='N')
    params.add_argument('--overwrite', type=cli.boolean, metavar='BOOL')

    # Determine default parameter values
    default_args = config.items('Default') \
        + config.items('Extraction') \
        + config.items('Training')
    default_args = [''] + cli.options_string(default_args)
    default_ns, _ = parser.parse_known_args(default_args)

    return parser.parse_args(remaining_args, default_ns)


if __name__ == '__main__':
    sys.exit(train(parse_args()))
