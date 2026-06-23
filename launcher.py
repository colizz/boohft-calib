"""
Launcher for the command line interface to the calibration tool. Example:
    python launcher.py cards/sfbdt/example_bb_PNetXbbVsQCD.yml

"""

from types import SimpleNamespace
import copy
import yaml
import ast
import os

from routines.sfbdt import SfbdtRoutine


ROUTINES = {
    'sfbdt': {
        'class': SfbdtRoutine,
        'card_dir': os.path.join('cards', 'sfbdt'),
    },
}


def resolve_config_path(config_path, routine):
    r"""Resolve card paths, including the pre-refactor cards/... layout."""

    if os.path.isfile(config_path):
        return config_path

    card_dir = ROUTINES[routine]['card_dir']
    if config_path.startswith('cards' + os.sep):
        fallback = os.path.join('cards', routine, os.path.relpath(config_path, 'cards'))
        if os.path.isfile(fallback):
            return fallback

    fallback = os.path.join(card_dir, os.path.basename(config_path))
    if os.path.isfile(fallback):
        return fallback

    raise FileNotFoundError(f'Cannot find config card: {config_path}')


def load_global_cfg(config_path, routine='sfbdt'):
    r"""Load the global configuration from the routine base card and custom card."""

    if routine not in ROUTINES:
        raise ValueError(f'Unsupported routine: {routine}. Available routines: {", ".join(sorted(ROUTINES))}')

    routine_cls = ROUTINES[routine]['class']
    config_path = resolve_config_path(config_path, routine)
    base_path = os.path.join(ROUTINES[routine]['card_dir'], 'base.yml')
    with open(base_path) as f:
        global_cfg = yaml.safe_load(f)
    with open(config_path) as f:
        global_cfg.update(yaml.safe_load(f))
    for subattr in getattr(routine_cls, 'config_namespaces', ()):
        global_cfg[subattr] = SimpleNamespace(**global_cfg[subattr])
    global_cfg['year'] = str(global_cfg['year'])
    global_cfg['routine'] = routine
    global_cfg['config_path'] = config_path
    global_cfg = SimpleNamespace(**global_cfg)
    return global_cfg


def launch(config_path, routine='sfbdt', workers=None, run_step=None, skip_coffea=None, options=None, multi_years=None):
    r"""Launch a selected calibration routine."""

    global_cfg_base = load_global_cfg(config_path, routine=routine)
    if workers is not None:
        global_cfg_base.workers = workers
    if run_step is not None:
        global_cfg_base.run_step = run_step
    if skip_coffea is not None:
        global_cfg_base.skip_coffea = skip_coffea
    if options is not None and len(options) > 0:
        options = {k: ast.literal_eval(v) for k, v in options}
        print(options)
        for k in options:
            setattr(global_cfg_base, k, options[k])

    routine_cls = ROUTINES[routine]['class']
    if multi_years is None:
        routine_cls(global_cfg_base).launch()
    else:
        assert all(year in ['2016APV', '2016', '2017', '2018'] for year in multi_years), "Please specify the correct year format"
        for year in multi_years:
            global_cfg = copy.deepcopy(global_cfg_base)
            global_cfg.year = year
            routine_cls(global_cfg).launch()


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser('Preprocess ntuples')
    parser.add_argument('config_path')
    parser.add_argument('--routine', '-r', default='sfbdt', choices=sorted(ROUTINES),
        help='Routine to run. Defaults to sfbdt.')
    parser.add_argument('--workers', '-w', nargs='+', type=int, default=None,
        help='Number of concurrent workers for the coffea and standalone processor. Will overide the option in base config.')
    parser.add_argument('--run-step', '-s', type=str, default=None,
        help='Bool digits to control whether or not to run each routine step. Will overide the option in base config.')
    parser.add_argument('--skip-coffea', action='store_true',
        help='If specified, skip running the coffea step and directly load the existing results (should guarantee that the coffea step has run before). '
             'Will overide the option in the base config.')
    parser.add_argument('--options', '-o', nargs=2, action='append', default=[],
        help='pass the options to override the original value in the YAML card')
    parser.add_argument('--multi-years', '-y', nargs='+', type=str, default=None,
        help='Specify one or multiple year(s) options to run. Will overide the option in the config card.')
    args = parser.parse_args()

    launch(
        args.config_path,
        routine=args.routine,
        workers=args.workers,
        run_step=args.run_step,
        skip_coffea=args.skip_coffea,
        options=args.options,
        multi_years=args.multi_years,
    )
