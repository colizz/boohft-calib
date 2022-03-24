This directory includes configuration cards for the recast taggers calibration, previously run on old versions of the sfBDT strategy.

Cards with `20220128_` correspond to the results from the [4 Feb (2022) BTV talk](https://indico.cern.ch/event/1120932/#23-calibration-of-ul20172018-x).

Commands run on PKU cluster:
```python
python launcher.py cards/recast/20220128_cc_preUL_recast_vhccBTV_PNetXccVsQCD_ak15.yml --workers 10 10 10 100 -y 2016 2017 2018
python launcher.py cards/recast/20220128_bb_preUL_recast_std_PNetXbbVsQCD_ak8.yml --workers 10 10 10 100 -y 2016 2017 2018
python launcher.py cards/recast/20220128_bb_UL_recast_XtoYHBTV_PNetXbbVsQCD_ak8.yml --workers 10 10 10 100 -y 2017 2018
```

Update from 17 March: UL recast with V-qq sample included:
```
python launcher.py cards/recast/20220317_bb_UL_recast_XtoYHBTVwVqqSample_PNetXbbVsQCD_ak8.yml --workers 10 10 10 100 -y 2017 2018
```