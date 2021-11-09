import argparse
import sys
from pathlib import Path

import cli


def predict(args):
    import pandas as pd

    import system.datasets as datasets
    import system.inference as inference

    # Load the relevant dataset metadata
    dataset = datasets.dataset(args.dataset, args.arca23k_dir,
                               args.fsd50k_dir, args.params.seed)
    subset = dataset[f'{args.subset}']

    # Generate predictions using several model checkpoints
    checkpoint_dir = args.work_dir / args.experiment_id / 'checkpoints'
    epochs = select_epochs(args.work_dir / args.experiment_id / 'logs')
    y_pred = inference.predict(subset, epochs, checkpoint_dir,
                               **vars(args.params))

    # Ensure output directory exists
    pred_dir = args.work_dir / args.experiment_id / 'predictions'
    pred_dir.mkdir(parents=True, exist_ok=True)

    # Save predictions to disk
    index = subset.tags.index
    if args.output_name:
        output_path = pred_dir / args.output_name
    else:
        output_path = pred_dir / f'{args.dataset.lower()}_{args.subset}.csv'
    df = pd.DataFrame(y_pred, index, dataset.label_set)
    df.to_csv(output_path)

    print(f'\nPredictions written to {output_path}')

    # Delete unused checkpoint files if applicable
    if args.clean:
        count = delete_checkpoints(checkpoint_dir, epochs)
        print(f'\nRemoved {count} unused checkpoint files')


def select_epochs(log_dir, metric='val_mAP', n_epochs=3, min_epoch=10):
    import pandas as pd

    df = pd.read_csv(log_dir / 'history.csv', index_col=0).iloc[min_epoch:]
    df.sort_values(by=metric, ascending=metric in ['val_loss'], inplace=True)
    return df.index.values[:n_epochs]


def delete_checkpoints(checkpoint_dir, exceptions):
    count = 0
    for path in checkpoint_dir.glob('checkpoint.[0-9][0-9].pth'):
        epoch = int(str(path)[-6:-4])
        if epoch not in exceptions:
            path.unlink()
            count += 1
    return count


def parse_args():
    config, conf_parser, remaining_args = cli.parse_config_args()
    parser = argparse.ArgumentParser(parents=[conf_parser])

    parser.add_argument('dataset', metavar='DATASET')
    parser.add_argument('subset', metavar='SUBSET')
    parser.add_argument('--experiment_id', metavar='ID')
    parser.add_argument('--work_dir', type=Path, metavar='DIR')
    parser.add_argument('--arca23k_dir', type=Path, metavar='DIR')
    parser.add_argument('--fsd50k_dir', type=Path, metavar='DIR')
    parser.add_argument('--output_name', metavar='FILE_NAME')
    parser.add_argument('--clean', type=cli.boolean, metavar='BOOL')

    # Group the following command-line parameters
    params = cli.GroupParser('params', parser)
    params.add_argument('--sample_rate', type=cli.maybe_float, metavar='NUM')
    params.add_argument('--block_length', type=cli.maybe_float, metavar='NUM')
    params.add_argument('--features', type=cli.dict, metavar='SPEC')
    params.add_argument('--cache_features', type=cli.boolean, metavar='BOOL')
    params.add_argument('--weights_path', type=Path, metavar='PATH')
    params.add_argument('--batch_size', type=int, metavar='N')
    params.add_argument('--partition', type=cli.dict, metavar='SPEC')
    params.add_argument('--n_workers', type=int, metavar='N')
    params.add_argument('--seed', type=cli.maybe_int, metavar='N')
    params.add_argument('--cuda', type=cli.boolean, metavar='BOOL')

    # Determine default parameter values
    default_args = config.items('Default') \
        + config.items('Extraction') \
        + config.items('Prediction')
    default_args = ['', ''] + cli.options_string(default_args)
    default_ns, _ = parser.parse_known_args(default_args)

    return parser.parse_args(remaining_args, default_ns)


if __name__ == '__main__':
    sys.exit(predict(parse_args()))
