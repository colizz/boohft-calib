# Tool for boosted heavy flavour jet tagger calibration

`boohft-calib` is a tool that serves for the data-MC calibration of the "boosted heavy flavour jet tagger" in CMS, based on the sfBDT coastline method.
The tool runs under the Run 2 UL condition using NanoAODv9.
It is designed for the calibration of any Xbb/Xcc type taggers composed of the branches in NanoAODv9. 

Users should specify in a data card the tagger expression, pre-defined WPs, etc., and a signal ROOT tree for extraction of the necessary signal tagger shape.
See details in the [example data card](cards/example_bb_PNetXbbVsQCD.yml) for calibrating the ParticleNet XbbVsQCD score.

The introduction of the method can be found in the [latest BTV slides](https://indico.cern.ch/event/1120932/#23-calibration-of-ul20172018-x).
Detailed documentation is provided in [AN-21-005](https://cms.cern.ch/iCMS/jsp/db_notes/noteInfo.jsp?cmsnoteid=CMS%20AN-2021/005).

## Run the tool

1. Run on a local cluster

First set up the environment. We recommand to use Miniconda:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ./miniconda  # for test: put the miniconda folder here
source miniconda/bin/activate
# install additional packages
pip install coffea uproot awkward lz4 xxhash numexpr PyYAML seaborn
# clone the repo
git clone https://github.com/colizz/boohft-calib.git && cd boohft-calib
```

Run the tool in one command, e.g.,
```bash
python launcher.py cards/example_bb_PNetXbbVsQCD.yml
```

See `launcher.py` for more information on the command arguments.

Note: the tool uses 8 concurrent workers by default. On lxplus it will run by estimation 4 hrs for an entire routine. Sepcify more workers if you have more CPU resource.

2. Run on SWAN

[![Open in SWAN](https://swanserver.web.cern.ch/swanserver/images/badge_swan_white_150.png)](https://cern.ch/swanserver/cgi-bin/go?projurl=https://github.com/colizz/boohft-calib.git)

Each routine will run by estimation 8 hrs on SWAN.

To run on SWAN, click the link, start a SWAN session with LCG96 Python3 stack (4 cores, 16GB), then open and run the `launcher_swan.ipynb` notebook. This will launch a routine configured by the example card.


## Update notes

v3.0.1 Jan 29 2022
 - Support more command line arguments

v3.0.0 Jan 24 2022
 - Update the method to sfBDT coastline
 - Update the framework to coffea (supports local run at present)

Previous version (till v2.1) developed in [`ParticleNet-CCTagCalib`](https://github.com/colizz/ParticleNet-CCTagCalib/)