#!/usr/bin/env python3

from __future__ import print_function
import argparse
import os
import shlex
import shutil
import subprocess

if "CMSSW_BASE" not in os.environ:
    raise RuntimeError("This script must run inside a CMSSW environment")

import CombineHarvester.CombineTools.ch as ch


parser = argparse.ArgumentParser("Run topwsf combine fit")
parser.add_argument("inputdir")
parser.add_argument("workdir")
parser.add_argument("--categories", required=True)
parser.add_argument("--processes", required=True)
parser.add_argument("--systematics", default="")
parser.add_argument("--year", required=True)
parser.add_argument("--lumi-uncertainty", type=float, required=True)
parser.add_argument("--auto-mc-stats", default="0")
parser.add_argument("--run-impact", action="store_true")
parser.add_argument("--impact-parallel", type=int, default=8)
args = parser.parse_args()


def runcmd(cmd):
    print(cmd)
    ret = subprocess.call(cmd, shell=True)
    return ret


os.makedirs(args.workdir, exist_ok=True)
for fname in ["inputs_pass.root", "inputs_fail.root"]:
    shutil.copy2(os.path.join(args.inputdir, fname), os.path.join(args.workdir, fname))
preview_card = os.path.join(args.inputdir, "datacard_preview.txt")
if os.path.isfile(preview_card):
    shutil.copy2(preview_card, os.path.join(args.workdir, "datacard_preview.txt"))

card = os.path.join(args.workdir, "SF.txt")
categories = [cat for cat in args.categories.split(",") if cat]
processes = [proc for proc in args.processes.split(",") if proc]
systematics = [syst for syst in args.systematics.split(",") if syst and syst != "nominal"]
if not categories:
    raise RuntimeError("No POI categories were configured")
if not processes:
    raise RuntimeError("No fit processes were configured")

cb = ch.CombineHarvester()
cb.SetVerbosity(1)

fit_cats = [(1, "pass"), (2, "fail")]
cb.AddObservations(["*"], [""], ["13TeV"], [""], fit_cats)

sig_procs = [categories[0]]
bkg_procs = [proc for proc in processes if proc not in sig_procs]
if bkg_procs:
    cb.AddProcesses(["*"], [""], ["13TeV"], [""], bkg_procs, fit_cats, False)
cb.AddProcesses(["*"], [""], ["13TeV"], [""], sig_procs, fit_cats, True)

for syst in systematics:
    if syst in ("jms", "jmr"):
        for proc in processes:
            cb.cp().process([proc]).AddSyst(cb, proc + syst, "shapeU", ch.SystMap()(1.0))
    else:
        cb.cp().process(processes).AddSyst(cb, syst, "shape", ch.SystMap()(1.0))

cb.cp().AddSyst(cb, "lumi_13TeV", "lnN", ch.SystMap()(args.lumi_uncertainty))
for proc in processes:
    if proc == "other":
        continue
    cb.cp().process([proc]).AddSyst(cb, proc + "_xsec", "lnN", ch.SystMap()(1.05))
if "other" in processes:
    cb.cp().process(["other"]).AddSyst(cb, "other_xsec", "lnU", ch.SystMap()(5.0))

for fit_bin in cb.bin_set():
    cb.cp().bin([fit_bin]).ExtractShapes(
        os.path.join(args.workdir, "inputs_%s.root" % fit_bin),
        "$PROCESS",
        "$PROCESS_$SYSTEMATIC",
    )

cb.PrintAll()
cb.WriteDatacard(card + ".tmp", os.path.join(args.workdir, "inputs_SF.root"))

with open(card, "w") as fout:
    with open(card + ".tmp") as fin:
        for line in fin:
            if "rateParam" in line:
                fout.write(line.replace("\n", "  [0.,10.]\n"))
            else:
                fout.write(line)
    for proc in ("tp3", "tp2", "tp1"):
        if proc in processes:
            fout.write("norm_top rateParam pass %s 1 [0.,10.]\n" % proc)
            fout.write("norm_top rateParam fail %s 1 [0.,10.]\n" % proc)
    fout.write("* autoMCStats %s\n" % args.auto_mc_stats)
os.remove(card + ".tmp")

ext_po = ""


def run_fit_once():
    cmd = f"""
cd {args.workdir} && \
echo "+++ Converting datacard to workspace +++" && \
text2workspace.py -m 125 -P HiggsAnalysis.CombinedLimit.TagAndProbeExtended:tagAndProbe SF.txt --PO categories={args.categories}{ext_po} > text2workspace.log 2>&1 && \
echo "+++ Running MultiDimFit +++" && \
combine -M MultiDimFit -m 125 SF.root --algo=singles --robustFit=1 --cminDefaultMinimizerTolerance 5. > fit.log 2>&1 && \
echo "+++ Running FitDiagnostics +++" && \
combine -M FitDiagnostics -m 125 SF.root --saveShapes --saveWithUncertainties --robustFit=1 --cminDefaultMinimizerTolerance 5. > fitdiagnostics.log 2>&1
"""
    return runcmd(cmd)


ret = run_fit_once()

fit_log = os.path.join(args.workdir, "fit.log")
fit_failed = ret != 0 or not os.path.isfile(fit_log) or "WARNING: MultiDimFit failed" in open(fit_log).read()
if fit_failed:
    raise RuntimeError("topwsf combine fit failed")

diff_nuisances = shutil.which("diffNuisances.py")
if diff_nuisances is None:
    candidate = os.path.join(os.environ["CMSSW_BASE"], "src", "HiggsAnalysis", "CombinedLimit", "test", "diffNuisances.py")
    if os.path.isfile(candidate):
        diff_nuisances = candidate

if diff_nuisances is None:
    with open(os.path.join(args.workdir, "diffnuisances.log"), "w") as f:
        f.write("diffNuisances.py was not found in PATH or CMSSW_BASE/src/HiggsAnalysis/CombinedLimit/test.\n")
else:
    diff_poi = "SF_" + categories[0]
    ret = runcmd(
        "cd {workdir} && python3 {script} fitDiagnosticsTest.root --poi {poi} -A -g diffnuisances.root > diffnuisances.log 2>&1".format(
            workdir=shlex.quote(args.workdir),
            script=shlex.quote(diff_nuisances),
            poi=shlex.quote(diff_poi),
        )
    )
    if ret != 0:
        with open(os.path.join(args.workdir, "diffnuisances.log"), "a") as f:
            f.write("\nERROR: diffNuisances.py failed for POI {poi} with exit code {ret}.\n".format(
                poi=diff_poi,
                ret=ret,
            ))

if args.run_impact:
    runcmd(f"""
cd {args.workdir} && \
combineTool.py -M Impacts -d SF.root -m 125 --doInitialFit --robustFit=1 > impacts.log 2>&1 && \
combineTool.py -M Impacts -d SF.root -m 125 --robustFit=1 --doFits --parallel {args.impact_parallel} >> impacts.log 2>&1 && \
combineTool.py -M Impacts -d SF.root -m 125 -o impacts.json >> impacts.log 2>&1 && \
plotImpacts.py -i impacts.json -o impacts >> impacts.log 2>&1
""")
