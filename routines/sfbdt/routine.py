import copy
import json
import os
import subprocess

from logger import _logger
from utils.routine_naming import routine_output_name
from utils.web_maker import WebMaker

from .mc_reweight_unit import MCReweightUnit
from .coastline_unit import CoastlineUnit
from .tmpl_writer_unit import TmplWriterUnit
from .fit_unit import FitUnit


def _current_commit_id():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


class SfbdtRoutine:
    name = "sfbdt"
    n_steps = 4
    config_namespaces = ("tagger", "main_analysis_tree")

    def __init__(self, global_cfg):
        self.global_cfg = global_cfg

    def _write_global_cfg(self, webpage):
        cfg_dump = copy.deepcopy(self.global_cfg.__dict__)
        for subattr in ['tagger', 'main_analysis_tree']:
            cfg_dump[subattr] = cfg_dump[subattr].__dict__
        web = WebMaker('global config')
        web.add_text('```json\n' + json.dumps(cfg_dump, indent=4) + '\n```')
        web.write_to_file(webpage, filename='global_cfg.html')

    def _write_navigation_webpage(self):
        job_name = routine_output_name(self.global_cfg)
        webpage = os.path.join('web', job_name)
        web = WebMaker(job_name)
        web.add_h1("Content")
        web.add_text(f"Results written by the `boohft-calib` framework {self.global_cfg.version}.")
        web.add_text(f"Current commit id is `{_current_commit_id()}`.")
        web.add_text()
        web.add_text(' 1. [MC reweighting](1_mc_reweight/): MC-to-data reweight plots.')
        web.add_text(' 2. [sfBDT coastline](2_coastline/): visualize the sfBDT coastline cut to make good gluon-spliting proxy.')
        web.add_text(' 3. [template writer](3_tmpl_writer/): Some inclusive plots on relavent variables.')
        web.add_text(' 4. [>> *SF summary* <<](4_fit/): Fit results and SF summary.')
        web.add_text()
        web.add_text('[Global config](global_cfg.html) for this routine.')
        web.write_to_file(webpage)
        self._write_global_cfg(webpage)
        return webpage

    def launch(self):
        _logger.info(f'Run with the global configuration: {self.global_cfg}')

        basedir = self.global_cfg.sample_prefix.replace('$YEAR', str(self.global_cfg.year))
        fileset = {
            sample: [os.path.join(basedir, relpath)]
            for sample, relpath in self.global_cfg.fileset_template.items()
        }
        workers = self.global_cfg.workers
        run_step = str(self.global_cfg.run_step)
        assert len(workers) == self.n_steps, "Invaild arguemnt for 'workers'."
        assert len(run_step) == self.n_steps and all(i in '01' for i in run_step), "Invaild arguemnt for 'run_step'."

        if run_step[0] == '1':
            if self.global_cfg.reuse_mc_weight_from_routine is None:
                _logger.info('Launch step 1: reweight total MC to data due to the use of prescaled HT triggers...')
                step_1 = MCReweightUnit(self.global_cfg, fileset=fileset, workers=workers[0])
                step_1.launch(skip_coffea=self.global_cfg.skip_coffea)
            else:
                _logger.info(f'Skip step 1 and reuse MC reweight factors from routine {self.global_cfg.reuse_mc_weight_from_routine}')

        if run_step[1] == '1':
            _logger.info('Launch step 2: calculate the sfBDT coastline on the target transformed tagger...')
            step_2 = CoastlineUnit(self.global_cfg, fileset=fileset, workers=workers[1])
            step_2.launch(skip_coffea=self.global_cfg.skip_coffea)

        if run_step[2] == '1':
            _logger.info('Launch step 3: derive the template for fit...')
            step_3 = TmplWriterUnit(self.global_cfg, fileset=fileset, workers=workers[2])
            step_3.launch(skip_coffea=self.global_cfg.skip_coffea)

        if run_step[3] == '1':
            _logger.info('Launch step 4: apply the fit to derive SFs, then make plots for the fit...')
            step_4 = FitUnit(self.global_cfg, fileset=fileset, workers=workers[3])
            step_4.launch(skip_coffea=self.global_cfg.skip_coffea)

        webpage = self._write_navigation_webpage()
        _logger.info(f'Job done! Everything write to webpage: {webpage}')
