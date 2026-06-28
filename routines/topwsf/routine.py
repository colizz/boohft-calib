import copy
import json
import os
import subprocess

from logger import _logger
from utils.routine_naming import routine_output_name
from utils.web_maker import WebMaker

from .templates_unit import TopWSFTemplatesUnit
from .fit_unit import TopWSFFitUnit


def _current_commit_id():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


class TopWSFRoutine:
    name = "topwsf"
    n_steps = 2
    config_namespaces = ("tagger",)

    def __init__(self, global_cfg):
        self.global_cfg = global_cfg

    def _write_global_cfg(self, webpage):
        cfg_dump = copy.deepcopy(self.global_cfg.__dict__)
        for subattr in self.config_namespaces:
            if subattr in cfg_dump:
                cfg_dump[subattr] = cfg_dump[subattr].__dict__
        web = WebMaker("global config")
        web.add_text("```json\n" + json.dumps(cfg_dump, indent=4) + "\n```")
        web.write_to_file(webpage, filename="global_cfg.html")

    def _write_navigation_webpage(self):
        job_name = routine_output_name(self.global_cfg)
        webpage = os.path.join("web", job_name)
        web = WebMaker(job_name)
        web.add_h1("Content")
        web.add_text(f"Results written by the `boohft-calib` framework {self.global_cfg.version}.")
        web.add_text(f"Current commit id is `{_current_commit_id()}`.")
        web.add_text()
        web.add_text(" 1. [Templates](1_templates/): coffea templates; MC/data inclusive, pT-binned, pass/fail histogram check; xsec weights and yield checks.")
        web.add_text(" 2. [>> *SF summary* <<](2_fit/): fit results, diagnostics, shape variations.")
        web.add_text()
        web.add_text("[Global config](global_cfg.html) for this routine.")
        web.write_to_file(webpage)
        self._write_global_cfg(webpage)
        return webpage

    def launch(self):
        _logger.info(f"Run topwsf with the global configuration: {self.global_cfg}")
        workers = self.global_cfg.workers
        run_step = str(self.global_cfg.run_step)
        assert len(workers) == self.n_steps, "Invalid argument for 'workers'."
        assert len(run_step) == self.n_steps and all(i in "01" for i in run_step), "Invalid argument for 'run_step'."

        if run_step[0] == "1":
            _logger.info("Launch topwsf step 1: derive 2D mass-pT templates.")
            step_1 = TopWSFTemplatesUnit(self.global_cfg, workers=workers[0])
            step_1.launch(skip_coffea=self.global_cfg.skip_coffea)

        if run_step[1] == "1":
            _logger.info("Launch topwsf step 2: write fit inputs and fit top/W tag-and-probe SFs.")
            step_2 = TopWSFFitUnit(self.global_cfg, workers=workers[1])
            step_2.launch(skip_coffea=self.global_cfg.skip_coffea)

        webpage = self._write_navigation_webpage()
        _logger.info(f"topwsf job done. Everything written to webpage: {webpage}")
