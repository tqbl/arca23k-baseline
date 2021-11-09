import argparse
import sys
from pathlib import Path

import cli


def evaluate(args):
    if args.output_name == '':
        args.output_name = f'{args.dataset.lower()}_{args.subset}.csv'

    if args.cached:
        scores = read_results(args)
    else:
        scores = evaluate_systems(args)

    print_results(scores)


def evaluate_systems(args):
    import pandas as pd

    import system.datasets as datasets
    import system.evaluation as evaluation

    # Load the relevant dataset metadata
    dataset = datasets.dataset(args.dataset, args.arca23k_dir, args.fsd50k_dir)
    subset = dataset[f'{args.subset}']

    scores = []
    for experiment_id in args.experiment_id:
        # Retrieve predictions and ground truth
        experiment_dir = args.work_dir / experiment_id
        pred_path = experiment_dir / 'predictions' / args.output_name
        y_pred = pd.read_csv(pred_path, index_col=0).values
        y_true = dataset.target(subset).values

        results = evaluation.evaluate(y_pred, y_true, dataset.label_set)

        # Ensure output directory exists and write results to disk
        result_dir = experiment_dir / 'results'
        result_dir.mkdir(parents=True, exist_ok=True)
        results.to_csv(result_dir / args.output_name)

        scores.append(results)
    return scores


def read_results(args):
    import pandas as pd

    scores = []
    for experiment_id in args.experiment_id:
        result_dir = args.work_dir / experiment_id / 'results'
        scores.append(pd.read_csv(result_dir / args.output_name, index_col=0))
    return scores


def print_results(scores):
    import pandas as pd

    pd.options.display.float_format = '{:,.4f}'.format
    pd.options.display.max_rows = 100

    if len(scores) == 1:
        scores = scores[0]
    else:
        scores = pd.concat(scores).loc['Micro Average']
        scores = pd.concat([scores.mean(), scores.sem() * 1.96],
                           axis=1, keys=['Mean', 'SEM'])

    print(str(scores))


def parse_args():
    config, conf_parser, remaining_args = cli.parse_config_args()
    parser = argparse.ArgumentParser(parents=[conf_parser])

    parser.add_argument('dataset', metavar='DATASET')
    parser.add_argument('subset', metavar='SUBSET')
    parser.add_argument('--experiment_id', type=cli.array, metavar='LIST')
    parser.add_argument('--work_dir', type=Path, metavar='DIR')
    parser.add_argument('--arca23k_dir', type=Path, metavar='DIR')
    parser.add_argument('--fsd50k_dir', type=Path, metavar='DIR')
    parser.add_argument('--output_name', metavar='FILE_NAME')
    parser.add_argument('--cached', type=cli.boolean, metavar='BOOL')

    default_args = dict(config.items('Default'))
    prediction_args = dict(config.items('Prediction'))
    parser.set_defaults(**default_args, **prediction_args)

    return parser.parse_args(remaining_args)


if __name__ == '__main__':
    sys.exit(evaluate(parse_args()))
