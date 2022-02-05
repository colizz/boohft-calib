"""
Launcher for the command line interface to the calibration tool. Example:
    python launcher.py cards/example_bb_PNetXbbVsQCD.yml

"""

from types import SimpleNamespace
import copy
import yaml
import json
import os

from mc_reweight_unit import MCReweightUnit
from coastline_unit import CoastlineUnit
from tmpl_writer_unit import TmplWriterUnit
from fit_unit import FitUnit
from utils.web_maker import WebMaker
from logger import _logger


def load_global_cfg(config_path):
    r"""Load the global configuration from the base config file and custom file"""

    with open('cards/base.yml') as f:
        global_cfg = yaml.safe_load(f)
    with open(config_path) as f:
        global_cfg.update(yaml.safe_load(f))
    for subattr in ['tagger', 'main_analysis_tree']:
        global_cfg[subattr] = SimpleNamespace(**global_cfg[subattr])
    global_cfg['year'] = str(global_cfg['year'])
    global_cfg = SimpleNamespace(**global_cfg)
    return global_cfg


def write_global_cfg(cfg, webpage):
    cfg_dump = copy.deepcopy(cfg.__dict__)
    for subattr in ['tagger', 'main_analysis_tree']:
        cfg_dump[subattr] = cfg_dump[subattr].__dict__
    web = WebMaker('global config')
    web.add_text('```json\n' + json.dumps(cfg_dump, indent=4) + '\n```')
    web.write_to_file(webpage, filename='global_cfg.html')


def launch_routine(global_cfg):

    _logger.info(f'Run with the global configuration: {global_cfg}')

    basedir = global_cfg.sample_prefix.replace('$YEAR', str(global_cfg.year))
    fileset = {sample: [os.path.join(basedir, relpath)] for sample, relpath in global_cfg.fileset_template.items()}
    workers = global_cfg.workers
    run_step = str(global_cfg.run_step)
    assert len(workers) == 4, "Invaild arguemnt for 'workers'."
    assert len(run_step) == 4 and all(i in '01' for i in run_step), "Invaild arguemnt for 'run_step'."

    # Run step 1-4 sequence
    if run_step[0] == '1':
        if global_cfg.reuse_mc_weight_from_routine is None:
            _logger.info('Launch step 1: reweight total MC to data due to the use of prescaled HT triggers...')
            step_1 = MCReweightUnit(global_cfg, fileset=fileset, workers=workers[0])
            step_1.launch()
        else:
            _logger.info(f'Skip step 1 and reuse MC reweight factors from routine {global_cfg.reuse_mc_weight_from_routine}')

    if run_step[1] == '1':
        _logger.info('Launch step 2: calculate the sfBDT coastline on the target transformed tagger...')
        step_2 = CoastlineUnit(global_cfg, fileset=fileset, workers=workers[1])
        step_2.launch()

    if run_step[2] == '1':
        _logger.info('Launch step 3: derive the template for fit...')
        step_3 = TmplWriterUnit(global_cfg, fileset=fileset, workers=workers[2])
        step_3.launch()
    
    if run_step[3] == '1':
        _logger.info('Launch step 4: apply the fit to derive SFs, then make plots for the fit...')
        step_4 = FitUnit(global_cfg, fileset=fileset, workers=workers[3])
        step_4.launch()

    # Make navigation webpage
    job_name = global_cfg.routine_name + '_' + str(global_cfg.year)
    webpage = os.path.join('web', job_name)
    web = WebMaker(job_name)
    web.add_h1("Content")
    web.add_text(f"Results written by the `boohft-calib` framework {global_cfg.version}.")
    web.add_text()
    web.add_text(' 1. [MC reweighting](1_mc_reweight/): MC-to-data reweight plots.')
    web.add_text(' 2. [sfBDT coastline](2_coastline/): visualize the sfBDT coastline cut to make good gluon-spliting proxy.')
    web.add_text(' 3. [template writer](3_tmpl_writer/): Some inclusive plots on relavent variables.')
    web.add_text(' 4. [>> *SF summary* <<](4_fit/): Fit results and SF summary.')
    web.add_text()
    web.add_text('[Global config](global_cfg.html) for this routine.')
    web.write_to_file(webpage)
    write_global_cfg(global_cfg, webpage)

    _logger.info(f'Job done! Everything write to webpage: {webpage}')


def launch(config_path, workers=None, run_step=None, multi_years=None):
    r"""Launch all steps"""

    global_cfg_base = load_global_cfg(config_path)
    if workers is not None:
        global_cfg_base.workers = workers
    if run_step is not None:
        global_cfg_base.run_step = run_step

    if multi_years is None: # use the year specified in the config card
        launch_routine(global_cfg_base)
    else:
        assert all(year in ['2016APV', '2016', '2017', '2018'] for year in multi_years), "Please specify the correct year format"
        for year in multi_years: # iterate over all specified year
            global_cfg = copy.deepcopy(global_cfg_base)
            global_cfg.year = year
            launch_routine(global_cfg)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser('Preprocess ntuples')
    parser.add_argument('config_path')
    parser.add_argument('--workers', '-w', nargs='+', type=int, default=None,
        help='Number of concurrent workers for the coffea and standalone processor. Will overide the option in base config.')
    parser.add_argument('--run-step', '-s', type=str, default=None,
        help='Four bool digits to control whether or not to run each of the four steps. Will overide the option in base config.')
    parser.add_argument('--multi-years', '-y', nargs='+', type=str, default=None,
        help='Specify one or multiple year(s) options to run. Will overide the option in the config card.')
    args = parser.parse_args()

    # Launch all steps
    launch(args.config_path, workers=args.workers, run_step=args.run_step, multi_years=args.multi_years)
