This directory includes configuration cards for the BTV custom study.

Cards with `20220130_` correspond to the results from the [4 Feb (2022) BTV talk](https://indico.cern.ch/event/1120932/#23-calibration-of-ul20172018-x).

Commands run on PKU cluster:
```python
python launcher.py cards/btv_custom/20220130_bb_ULNanoV9_PNetXbbVsQCD_ak8_2017.yml --workers 10 10 10 100
python launcher.py cards/btv_custom/20220130_bb_ULNanoV9_PNetXccVsQCD_ak8_2017.yml --workers 10 10 10 100
python launcher.py cards/btv_custom/20220130_bb_ULNanoV9_DDBvLV2_ak8_2017.yml --workers 10 10 10 100
python launcher.py cards/btv_custom/20220130_bb_ULNanoV9_DDCvLV2_ak8_2017.yml --workers 10 10 10 100
python launcher.py cards/btv_custom/20220130_bb_ULNanoV9_PNetXbbVsQCD_ak8_2018.yml --workers 10 10 10 100
python launcher.py cards/btv_custom/20220130_bb_ULNanoV9_PNetXccVsQCD_ak8_2018.yml --workers 10 10 10 100
python launcher.py cards/btv_custom/20220130_bb_ULNanoV9_DDBvLV2_ak8_2018.yml --workers 10 10 10 100
python launcher.py cards/btv_custom/20220130_bb_ULNanoV9_DDCvLV2_ak8_2018.yml --workers 10 10 10 100
```