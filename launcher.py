"""
Launcher for the command line interface to the calibration tool. Example:
    python launcher.py cards/config_bb_PNetXbbVsQCD.yml --worker 8 8 8 8

"""

from types import SimpleNamespace
import yaml
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
    global_cfg = SimpleNamespace(**global_cfg)
    return global_cfg


def launch(config_path, workers=[8, 8, 8, 8]):
    r"""Launch all steps"""

    global_cfg = load_global_cfg(config_path)

    if isinstance(workers, int):
        workers = [workers] * 4

    basedir = global_cfg.sample_prefix.replace('$YEAR', str(global_cfg.year))
    fileset = {
        'qcd-mg':     [os.path.join(basedir, 'mc/qcd-mg_tree.root')],
        'top':        [os.path.join(basedir, 'mc/top_tree.root')],
        'v-qq':       [os.path.join(basedir, 'mc/v-qq_tree.root')],
        'jetht':      [os.path.join(basedir, 'data/jetht_tree.root')],
    }

    # Run step 1-4 sequence
    if global_cfg.reuse_mc_weight_from_routine is None:
        _logger.info('Launch step 1: reweight total MC to data due to the use of prescaled HT triggers...')
        step_1 = MCReweightUnit(global_cfg, fileset=fileset, workers=workers[0])
        step_1.launch()
    else:
        _logger.info(f'Skip step 1 and reuse MC reweight factors from routine {global_cfg.reuse_mc_weight_from_routine}')

    _logger.info('Launch step 2: calculate the sfBDT coastline on the target transformed tagger...')
    step_2 = CoastlineUnit(global_cfg, fileset=fileset, workers=workers[1])
    step_2.launch()

    _logger.info('Launch step 3: derive the template for fit...')
    step_3 = TmplWriterUnit(global_cfg, fileset=fileset, workers=workers[2])
    step_3.launch()
    
    _logger.info('Launch step 4: apply the fit to derive SFs, then make plots for the fit...')
    step_4 = FitUnit(global_cfg, fileset=fileset, workers=workers[3])
    step_4.launch()

    # Make navigation webpage
    job_name = global_cfg.routine_name + '_' + str(global_cfg.year)
    webpage = os.path.join('web', job_name)
    web = WebMaker(job_name)
    web.add_h1("Content")
    web.add_text("Results written by the `cms-hrt-calib` framework v3-0 beta.")
    web.add_text()
    web.add_text(' 1. [MC reweighting](1_mc_reweight/): MC-to-data reweight plots.')
    web.add_text(' 2. [sfBDT coastline_unit](2_coastline/): visualize the sfBDT coastline_unit cut to make good gluon-spliting proxy.')
    web.add_text(' 3. [template writer](3_tmpl_writer/): Some inclusive plots on relavent variables.')
    web.add_text(' 4. [>> *SF summary* <<](4_fit/): Fit results and SF summary.')
    web.write_to_file(webpage)
    _logger.info(f'Job done! Everything write to webpage: {webpage}')


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser('Preprocess ntuples')
    parser.add_argument('config_path')
    parser.add_argument('--workers', '-w', type=list, default=[8, 8, 8, 8], 
        help='Number of concurrent workers for the coffea and standalone processor')
    args = parser.parse_args()

    # Launch all steps
    launch(args.config_path)
