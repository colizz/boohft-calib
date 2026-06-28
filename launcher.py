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
from routines.topwsf import TopWSFRoutine


ROUTINES = {
    'sfbdt': {
        'class': SfbdtRoutine,
        'card_dir': os.path.join('cards', 'sfbdt'),
    },
    'topwsf': {
        'class': TopWSFRoutine,
        'card_dir': os.path.join('cards', 'topwsf'),
    },
}

VALID_YEARS = [
    '2016APV', '2016', '2017', '2018',
    '2022', '2022EE', '2023', '2023BPix',
    '2022Comb', '2023Comb', '2024', '2025',
]


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


def _merge_cfg(base, update):
    r"""Merge one YAML payload into another using top-level replacement."""

    if update is None:
        return base
    update = dict(update)
    update.pop('extends', None)
    base.update(update)
    return base


def _resolve_extend_path(parent_path, extend_path, routine):
    r"""Resolve an extends entry relative to the card that declares it."""

    if os.path.isabs(extend_path):
        return extend_path

    parent_dir = os.path.dirname(parent_path)
    candidate = os.path.normpath(os.path.join(parent_dir, extend_path))
    if os.path.isfile(candidate):
        return candidate

    candidate = resolve_config_path(extend_path, routine)
    if os.path.isfile(candidate):
        return candidate

    raise FileNotFoundError(f'Cannot find extended config card: {extend_path}')


def _load_yaml_with_extends(config_path, routine, stack=None):
    r"""Load a YAML card and all cards listed in its extends field."""

    config_path = resolve_config_path(config_path, routine)
    if stack is None:
        stack = []
    if config_path in stack:
        cycle = ' -> '.join(stack + [config_path])
        raise RuntimeError(f'Circular YAML extends detected: {cycle}')

    with open(config_path) as f:
        payload = yaml.safe_load(f) or {}

    merged = {}
    extends = payload.get('extends', [])
    if isinstance(extends, str):
        extends = [extends]
    for extend_path in extends:
        extend_path = _resolve_extend_path(config_path, extend_path, routine)
        merged = _merge_cfg(merged, _load_yaml_with_extends(extend_path, routine, stack + [config_path]))
    return _merge_cfg(merged, payload)


def _load_yaml_sequence(config_paths, routine):
    r"""Load and merge one card or a list of cards in order."""

    if isinstance(config_paths, (list, tuple)):
        merged = {}
        for config_path in config_paths:
            merged = _merge_cfg(merged, _load_yaml_with_extends(config_path, routine))
        return merged
    return _load_yaml_with_extends(config_paths, routine)


def load_global_cfg(config_path, routine='sfbdt'):
    r"""Load the global configuration from the routine base card and custom card."""

    if routine not in ROUTINES:
        raise ValueError(f'Unsupported routine: {routine}. Available routines: {", ".join(sorted(ROUTINES))}')

    routine_cls = ROUTINES[routine]['class']
    resolved_config_path = resolve_config_path(config_path[0] if isinstance(config_path, (list, tuple)) else config_path, routine)
    base_path = os.path.join(ROUTINES[routine]['card_dir'], 'base.yml')
    with open(resolved_config_path) as f:
        payload = yaml.safe_load(f) or {}
    if 'extends' in payload or isinstance(config_path, (list, tuple)):
        global_cfg = _load_yaml_sequence(config_path, routine)
    else:
        with open(base_path) as f:
            global_cfg = yaml.safe_load(f)
        global_cfg = _merge_cfg(global_cfg, payload)
    for subattr in getattr(routine_cls, 'config_namespaces', ()):
        global_cfg[subattr] = SimpleNamespace(**global_cfg[subattr])
    global_cfg['year'] = str(global_cfg['year'])
    global_cfg['routine'] = routine
    global_cfg['config_path'] = resolved_config_path
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
        assert all(year in VALID_YEARS for year in multi_years), "Please specify the correct year format"
        for year in multi_years:
            global_cfg = copy.deepcopy(global_cfg_base)
            global_cfg.year = year
            routine_cls(global_cfg).launch()


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser('Preprocess ntuples')
    parser.add_argument('config_path', nargs='+')
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
        args.config_path[0] if len(args.config_path) == 1 else args.config_path,
        routine=args.routine,
        workers=args.workers,
        run_step=args.run_step,
        skip_coffea=args.skip_coffea,
        options=args.options,
        multi_years=args.multi_years,
    )
