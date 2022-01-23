# Tool for boosted heavy flavour jet tagger calibration

`boohft-calib` is a tool that serves for the data-MC calibration of the "boosted heavy flavour jet tagger" in CMS, based on the sfBDT coastline method.
The tool runs under the Run 2 UL condition using NanoAODv9.
It is designed for the calibration of any Xbb/Xcc type taggers composed of the branches in NanoAODv9. 

Users should specify in a data card the tagger expression, pre-defined WPs, etc., and a signal ROOT tree for extraction of the necessary signal tagger shape.
See details in the [example data card]().

The introduction of the method can be found in [these slides](TBA).
Detailed documentation is provided in [AN-21-005](https://cms.cern.ch/iCMS/jsp/db_notes/noteInfo.jsp?cmsnoteid=CMS%20AN-2021/005).

## Update notes

v3.0.0 Jan 24 2022
 - Update the method to sfBDT coastline
 - Update the framework to coffea (supports local run at present)

Previous version (till v2.1) developed in [`ParticleNet-CCTagCalib`](https://github.com/colizz/ParticleNet-CCTagCalib/)