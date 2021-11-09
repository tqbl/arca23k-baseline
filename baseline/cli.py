import argparse
import ast
import configparser
from pathlib import Path


class GroupParser:
    def __init__(self, name, parent):
        self.name = name
        self.parent = parent

    def add_argument(self, name, **kwargs):
        class _GroupAction(argparse.Action):
            def __call__(nself, parser, namespace, values, option_string=None):
                ns = getattr(namespace, self.name, argparse.Namespace())
                setattr(ns, nself.dest, values)
                setattr(namespace, self.name, ns)

        self.parent.add_argument(name, default=argparse.SUPPRESS,
                                 action=_GroupAction, **kwargs)


def parse_config_args():
    conf_parser = argparse.ArgumentParser(add_help=False)
    conf_parser.add_argument('-f', '--config_file', type=Path,
                             action='append', metavar='FILE')
    args, remaining_args = conf_parser.parse_known_args()

    # Parse the config file(s). The default config file is a fallback
    # for options that are not specified by the user.
    config = configparser.ConfigParser()
    try:
        config.read_file(open('default.ini'))
    except FileNotFoundError:
        raise FileNotFoundError('default.ini is missing!')

    if args.config_file:
        for path in args.config_file:
            try:
                config.read_file(open(path))
            except FileNotFoundError:
                raise FileNotFoundError(f'Config file not found: {str(path)}')

    return config, conf_parser, remaining_args


def options_string(args):
    return [s for pair in args for s in ['--' + pair[0], pair[1]]]


def boolean(arg):
    if arg.lower() == 'true':
        return True
    if arg.lower() == 'false':
        return False
    raise argparse.ArgumentTypeError('boolean value expected')


def maybe_int(arg):
    return int(arg) if arg != '' else None


def maybe_float(arg):
    return float(arg) if arg != '' else None


def array(arg):
    return arg.replace(' ', '').split(',')


def dict(arg):
    return ast.literal_eval(arg) if arg else {}
