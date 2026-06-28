# boohft-calib

`boohft-calib` is a CMS calibration framework for data-MC scale factors of
boosted heavy-flavour jet taggers.

The previous generation of this package was designed around the sfBDT method
for X->bb tagging calibration; see the
[`release/v3` branch](https://github.com/colizz/boohft-calib/tree/release/v3).
The current framework is being generalized to support multiple taggers,
calibration phase spaces, and fitting strategies through routine-specific YAML
cards.

## Routines

- [`sfbdt`](routines/sfbdt): calibration with the sfBDT method in a
  heavy-flavour enriched QCD phase space. The method uses sfBDT-selected
  g->bb/cc proxy jets to calibrate X->bb/cc jets. For the method details, see
  the [BTV-22-001 paper](https://arxiv.org/abs/2510.10228) or
  [AN-21-005](https://cms.cern.ch/iCMS/jsp/db_notes/noteInfo.jsp?cmsnoteid=CMS%20AN-2021/005)
  (`sfBDT method`).

- [`topwsf`](routines/topwsf): top/W tagger calibration in a semileptonic ttbar
  phase space, using generator-matched top-merged and W-merged jets. The
  workflow is a modern-framework rewrite of
  [`ParticleNetSF`](https://github.com/cms-jet/ParticleNetSF); the processing and
  fit logic follow the same strategy.

- `zbb`: X->bb tagger calibration from the Z->bb peak in a QCD phase space,
  with a dimuon Z->mumu phase space used to constrain the inclusive Z+jets
  cross section. This routine is still <span style="color:red">under development</span>.

## Framework Philosophy

The main goal of `boohft-calib` is to make custom scale-factor derivation
practical for analyzers while keeping the intermediate checks reviewable.

Each routine is piloted by a YAML card. After the routine finishes, the
calibration results, final plots, and intermediate diagnostic plots are
collected into automatically generated webpages. This is useful for checking
histograms, templates, fit quality, nuisance impacts, and yield bookkeeping,
and it is particularly helpful when an analysis needs custom SFs that still
require POG review.

Since the v4 series, the framework no longer requires a local conda
environment. The standard workflow runs directly from the CERN LCG stack, for
example (on LXPLUS):

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc15-opt/setup.sh
```

For the original sfBDT-driven webpage workflow and the early framework
motivation, see this
[this presentation slide](https://indico.cern.ch/event/1120932/contributions/4706547/attachments/2384057/4074342/22.02.03_BTV_Calibration%20of%20UL2017_2018%20Xbb_cc%20taggers%20with%20sfBDT%20coastline%20method.pdf#page=20)
from the first public release.

## Running

Routine-specific instructions are documented in:

- [`routines/sfbdt`](routines/sfbdt)
- [`routines/topwsf`](routines/topwsf)

Try `python launcher.py --help` for the common launcher options.

## Update Notes

v4.0 June 28, 2026
- Major refactor from an sfBDT-only package into a multi-routine calibration
  framework.
- Add the `topwsf` routine for top/W tagger calibration.
- Move routine-specific documentation and cards into routine-oriented
  workflows.
- Use the LCG software stack as the default runtime environment.

### Early Updates for v3

v3.1.3 November 8, 2024
- Feature: allow setting an individual fit range for the main POI.
- Feature: allow customisation of sfBDT input variables.
- Update: adapt code compatibility to the EL9 system.
- Update: reduce the default numbers of parallel workers from 8 to 5 to prevent
  warnings on lxplus.

v3.1.2 July 21, 2023
- Update: fix lumi uncertainty.
- Update: apply no JERC correction to SV mass.

v3.1.1 May 25, 2023
- Update: change the 20% frac_b/c/light variation in an overall manner, synced
  with the mu-tagged method.
- Update: in case of a fit failure, enlarge the autoMCStats threshold and
  retry.
- Feature: more text on plots to improve readability.

v3.1.0 December 2, 2022
- Feature: add new uncertainty sources.
- Feature: allow breaking down the full uncertainty list.

v3.0.5 November 25, 2022
- Feature: allow using custom sfBDT models to replace the default one.

v3.0.4 April 19, 2022
- Feature improved: allow expression parsing with awkward-array indexing.
- Reweight binning bug fix.

v3.0.3 March 31, 2022
- Implement the qq calibration type.

v3.0.2 February 5, 2022
- Implement the year condition for 2016APV and 2016.

v3.0.1 January 29, 2022
- Support more command-line arguments.

v3.0.0 January 24, 2022
- Update the method to sfBDT coastline.
- Update the framework to coffea.

### Early Updates for v2

Previous versions through v2.1 were developed in
[`ParticleNet-CCTagCalib`](https://github.com/colizz/ParticleNet-CCTagCalib/).
