#!/usr/bin/env python

# first ensure that the script runs in the CMSSW environment
import os
if 'CMSSW_BASE' not in os.environ:
    raise Exception("The scirpt need to be run in the CMSSW environment")

import CombineHarvester.CombineTools.ch as ch
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
import os

import argparse
parser = argparse.ArgumentParser('Preprocess ntuples')
parser.add_argument('inputdir', help='Input directory')
parser.add_argument('workdir', help='Working directory for the fit')
parser.add_argument('--type', default=None, choices=['bb', 'cc', 'qq'], help='bb, cc, or qq calibration type')
parser.add_argument('--mode', default=None, choices=['main', 'sfbdt_rwgt', 'fit_var_rwgt'], help='fit schemes in controlling the nuiences.')
parser.add_argument('--ext-unce', default=None, help='Extra uncertainty term to run or term to be excluded. e.g. --ext-unce NewTerm1,NewTerm2,~ExcludeTerm1')
parser.add_argument('--bound', default=None, help='Set the bound of three SFs, e.g. --bound 0.5,2 (which are the default values)')
parser.add_argument('--run-impact', action='store_true', help='Run impact plots.')
parser.add_argument('--run-unce-breakdown', action='store_true', help='Run uncertainty breakdown')
args = parser.parse_args()

if args.type == 'bb':
    flv_poi1, flv_poi2, flv_poi3 = 'flvB', 'flvC', 'flvL'
elif args.type == 'cc':
    flv_poi1, flv_poi2, flv_poi3 = 'flvC', 'flvB', 'flvL'
elif args.type == 'qq':
    flv_poi1, flv_poi2, flv_poi3 = 'flvL', 'flvB', 'flvC'

if not os.path.exists(args.workdir):
    os.makedirs(args.workdir)
        
cb = ch.CombineHarvester()
cb.SetVerbosity(1)

useAutoMCStats = True
inputdir = args.inputdir
outputname = 'SF.txt'

cats = [
    (1, 'pass'),
    (2, 'fail'),
    ]

cb.AddObservations(['*'], [''], ['13TeV'], [''], cats)

bkg_procs = [flv_poi2, flv_poi3]
cb.AddProcesses(['*'], [''], ['13TeV'], [''], bkg_procs, cats, False)

sig_procs = [flv_poi1]
cb.AddProcesses(['*'], [''], ['13TeV'], [''], sig_procs, cats, True)

all_procs = sig_procs + bkg_procs

bins = cb.bin_set()

shapeSysts = {
    'pu':all_procs,
    'fracBB':['flvB'],
    'fracCC':['flvC'],
    'fracLight':['flvL'],
    'psWeightIsr':all_procs,
    'psWeightFsr':all_procs,
    'sfBDTRwgt':all_procs,
    'fitVarRwgt':all_procs,
}
if args.ext_unce is not None:
    for ext_unce in args.ext_unce.split(','):
        if not ext_unce.startswith('~'):
            shapeSysts[ext_unce] = all_procs
        else:
            shapeSysts.pop(ext_unce[1:], None)

for syst in shapeSysts:
    cb.cp().process(shapeSysts[syst]).AddSyst(cb, syst, 'shape', ch.SystMap()(1.0))

cb.cp().AddSyst(cb, 'lumi_13TeV', 'lnN', ch.SystMap()(1.025))

for bin in bins:
    cb.cp().bin([bin]).ExtractShapes(
        os.path.join(args.inputdir, 'inputs_%s.root' % bin),
        '$PROCESS',
        '$PROCESS_$SYSTEMATIC'
        )

if not useAutoMCStats:
    bbb = ch.BinByBinFactory()
    bbb.SetAddThreshold(0.1).SetFixNorm(True)
    bbb.AddBinByBin(cb.cp().backgrounds(), cb)

cb.PrintAll()

outputroot = os.path.join(args.workdir, 'inputs_%s.root' % outputname.split('.')[0])
cb.WriteDatacard(os.path.join(args.workdir, outputname + '.tmp'), outputroot)

with open(os.path.join(args.workdir, outputname), 'w') as fout:
    with open(os.path.join(args.workdir, outputname + '.tmp')) as f:
        for l in f:
            if 'rateParam' in l:
                fout.write(l.replace('\n', '  [0.2,5]\n'))
            else:
                fout.write(l)
    os.remove(os.path.join(args.workdir, outputname + '.tmp'))

    if useAutoMCStats:
        fout.write('* autoMCStats 20\n')

## Start running higgs combine

import subprocess
def runcmd(cmd, shell=True):
    """Run a shell command"""
    p = subprocess.Popen(
        cmd, shell=shell, universal_newlines=True
    )
    out, _ = p.communicate()
    return (out, p.returncode)

ext_po = '' if args.bound is None else '--PO bound='+args.bound
if args.mode == 'main':
    ext_fit_options = '--setParameters sfBDTRwgt=0,fitVarRwgt=0 --freezeParameters sfBDTRwgt,fitVarRwgt'
elif args.mode == 'sfbdt_rwgt':
    ext_fit_options = '--setParameters sfBDTRwgt=1,fitVarRwgt=0 --freezeParameters sfBDTRwgt,fitVarRwgt'
elif args.mode == 'fit_var_rwgt':
    ext_fit_options = '--setParameters sfBDTRwgt=0,fitVarRwgt=1 --freezeParameters sfBDTRwgt,fitVarRwgt'

runcmd('''
cd {workdir} && \
echo "+++ Converting datacard to workspace +++" && \
text2workspace.py -m 125 -P HiggsAnalysis.CombinedLimit.TagAndProbeExtendedV2:tagAndProbe SF.txt --PO categories={flv_poi1},{flv_poi2},{flv_poi3} {ext_po} && \
echo "+++ Fitting... +++" && \
combine -M MultiDimFit -m 125 SF.root --algo=singles --robustFit=1 {ext_fit_options} > fit.log && \
combine -M FitDiagnostics -m 125 SF.root --saveShapes --saveWithUncertainties --robustFit=1 {ext_fit_options} > /dev/null 2>&1
'''.format(workdir=args.workdir, flv_poi1=flv_poi1, flv_poi2=flv_poi2, flv_poi3=flv_poi3, ext_po=ext_po, ext_fit_options=ext_fit_options)
)

if args.run_impact:
    runcmd('''
cd {workdir} && \
combineTool.py -M Impacts -d SF.root -m 125 --doInitialFit --robustFit=1 {ext_fit_options} > pdf.log 2>&1 && \
combineTool.py -M Impacts -d SF.root -m 125 --robustFit=1 --doFits {ext_fit_options} >> pdf.log 2>&1 && \
combineTool.py -M Impacts -d SF.root -m 125 -o impacts.json {ext_fit_options} >> pdf.log 2>&1 && \
plotImpacts.py -i impacts.json -o impacts >> pdf.log 2>&1
'''.format(workdir=args.workdir, ext_fit_options=ext_fit_options)
    )

if args.run_unce_breakdown:
    runcmd('''
cd {workdir} && \
combine -M MultiDimFit -m 125 SF.root --algo=grid --robustFit=1 --points=50 -n Grid --redefineSignalPOIs SF_{flv_poi1} {ext_fit_options} && \
plot1DScan.py higgsCombineGrid.MultiDimFit.mH125.root --POI SF_{flv_poi1} && \
combine -M MultiDimFit -m 125 SF.root --algo=singles --robustFit=1 -n Bestfit --saveWorkspace {ext_fit_options} && \
combine -M MultiDimFit -m 125 --algo=grid --points=50 -n Stat higgsCombineBestfit.MultiDimFit.mH125.root --redefineSignalPOIs SF_{flv_poi1} --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances && \
plot1DScan.py higgsCombineGrid.MultiDimFit.mH125.root --others 'higgsCombineStat.MultiDimFit.mH125.root:FreezeAll:2' --POI SF_{flv_poi1} -o unce_breakdown --breakdown Syst,Stat
'''.format(workdir=args.workdir, flv_poi1=flv_poi1, ext_fit_options=ext_fit_options)
    )