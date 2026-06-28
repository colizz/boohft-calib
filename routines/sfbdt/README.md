# sfbdt routine

The `sfbdt` routine calibrates X->bb/cc taggers with the sfBDT coastline method
in a heavy-flavour enriched QCD phase space. It is the original workflow of
`boohft-calib` and is documented in
[AN-21-005](https://cms.cern.ch/iCMS/jsp/db_notes/noteInfo.jsp?cmsnoteid=CMS%20AN-2021/005).

## Example routine

Set up the LCG environment (e.g., for LXPLUS):

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc15-opt/setup.sh
```

Run the sfBDT example card:

```bash
python launcher.py cards/sfbdt/example_bb_PNetXbbVsQCD.yml --routine sfbdt --workers 10 10 10 70
```

`sfbdt` is currently the default routine, so `--routine sfbdt` may be omitted.

The four-step routine can also be run one step at a time:

```bash
python launcher.py cards/sfbdt/example_bb_PNetXbbVsQCD.yml --routine sfbdt --workers 10 10 10 70 -s 1000
python launcher.py cards/sfbdt/example_bb_PNetXbbVsQCD.yml --routine sfbdt --workers 10 10 10 70 -s 0100
python launcher.py cards/sfbdt/example_bb_PNetXbbVsQCD.yml --routine sfbdt --workers 10 10 10 70 -s 0010
python launcher.py cards/sfbdt/example_bb_PNetXbbVsQCD.yml --routine sfbdt --workers 10 10 10 70 -s 0001
```

The example routine has been validated under LCG_110 for steps 1-3 with the
commands above. Step 4 depends on the fit workflow and is not included in this
LCG_110 validation note.

Try `python launcher.py --help` for common launcher options.

## SWAN

[![Open in SWAN](https://swanserver.web.cern.ch/swanserver/images/badge_swan_white_150.png)](https://cern.ch/swanserver/cgi-bin/go?projurl=https://github.com/colizz/boohft-calib.git)

To run on SWAN, click the link, start a SWAN session with the LCG_110 Python
stack or source the same LCG view in the notebook, then open
`launcher_swan.ipynb` and run all blocks. This launches a routine configured by
the example card.

## Configuration card

The configuration card, for example
[`cards/sfbdt/example_bb_PNetXbbVsQCD.yml`](../../cards/sfbdt/example_bb_PNetXbbVsQCD.yml),
defines the full routine configuration. Users should specify:

- the calibration type: `bb`, `cc`, or `qq`;
- the UL data-taking year: `2016APV`, `2016`, `2017`, or `2018`;
- jet pT ranges for deriving separate SFs;
- tagger information, including the tagger name/expression, score span, and
  custom WPs used in the analysis;
- a signal ROOT tree from the analysis, used to extract the signal tagger
  shape.

Detailed comments are provided directly in the example card.
