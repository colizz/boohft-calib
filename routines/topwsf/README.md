# topwsf routine

## Example routine

Set up the LCG environment (e.g., for LXPLUS):

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc15-opt/setup.sh
```

Run all configured steps:

```bash
python launcher.py cards/topwsf/example_topwsf_w_PNetMD_2018_v9.yml --routine topwsf --run-step 11 --workers 20 20
```

Step outputs are written to `output/{routine_name}_{year}_{nano_version}/`. Web pages are written to `web/{routine_name}_{year}_{nano_version}/`.

The routine has two steps:

```text
1_templates: coffea templates, validation plots, xsec weights, and yield checks
2_fit: projected ROOT inputs, preview cards, shape variations, combine fits, diagnostics, impacts, and SF summaries
```

### Tips

Split steps with the `--run-step` bit mask:

```bash
python launcher.py cards/topwsf/example_topwsf_w_PNetMD_2018_v9.yml --routine topwsf --run-step 10 --workers 20 20
python launcher.py cards/topwsf/example_topwsf_w_PNetMD_2018_v9.yml --routine topwsf --run-step 01 --workers 20 20
```

Use `--skip-coffea` when rerunning step 1 webpages or plots from an existing step-1 `result.pickle`/`templates2d.pickle` without reprocessing ntuples:

```bash
python launcher.py cards/topwsf/example_topwsf_w_PNetMD_2018_v9.yml --routine topwsf --run-step 10 --skip-coffea --workers 20 20
```

Use `skip_fit: true` in the card, or override it from the command line, when combine outputs already exist and only the fit webpage/plots need to be regenerated:

```bash
python launcher.py cards/topwsf/example_topwsf_w_PNetMD_2018_v9.yml --routine topwsf --run-step 01 --workers 20 20 -o skip_fit True
```

Add QCD MG samples by enabling the `qcd-mg` group. They enter the `other` fit process unless the process model is changed:

```bash
python launcher.py cards/topwsf/example_topwsf_w_PNetMD_2018_v9.yml \
  --routine topwsf --run-step 11 --workers 20 20 \
  -o routine_name '"example_topwsf_w_PNetMD_withqcd"' \
  -o enabled_sample_groups '["ttbar-powheg", "singletop", "diboson", "ttv", "w", "qcd-mg"]'
```

## Background-efficiency WP scan

`scan_wp.py` is a standalone helper. It is not part of the routine step chain and only prints results to the terminal.

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc15-opt/setup.sh
python routines/topwsf/scan_wp.py cards/topwsf/example_topwsf_w_PNetMD_2018_v9.yml
```

The script reads the YAML card, uses the configured `tagger.expr` and `tagger.type`, and scans nominal MC only. The hard-coded target background efficiencies are:

```text
0.05, 0.025, 0.01, 0.005, 0.001
```

For `tagger.type: w`, `tp2` is treated as signal and all other MC processes are background. For `tagger.type: top`, `tp3` is treated as signal and all other MC processes are background.

The scan applies the configured base selection plus the template mass and pT ranges from `template_mass_bins` and `template_pt_bins`. It uses nominal event weights:

```text
genWeight * xsecWeight * puWeight * lumi
```

and also applies `topptWeight` when `apply_toppt_weight` is true and the branch is present.
